import torch
import torch.nn.functional as F
import logging
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, cohen_kappa_score,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device,
                 class_names: list, weights_path: str, architecture: str = "convnext_tiny",
                 mode: str = "species_only", scheduler=None, patience: int = 10,
                 lambda_sex: float = 0.3, lambda_form: float = 0.2):
        self.model        = model.to(device)
        self.dataloaders  = dataloaders
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.device       = device
        self.class_names  = class_names
        self.architecture = architecture
        self.mode         = mode
        self.scheduler    = scheduler
        self.patience     = patience
        self.lambda_sex   = lambda_sex
        self.lambda_form  = lambda_form

        self.counter      = 0
        self.best_val_acc = 0.0

        # modes that receive 4-element batches from MultiTaskDataset
        self._uses_aux = mode in ("multi_task", "sex_only", "form_only")

        self.save_path = Path(weights_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(self, num_epochs: int):
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_loss, train_acc = self._run_epoch("train")
            val_loss,   val_acc   = self._run_epoch("val")

            logger.info(
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

            mlflow.log_metrics({
                "train_loss":    train_loss,
                "train_acc":     train_acc,
                "val_loss":      val_loss,
                "val_acc":       val_acc,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }, step=epoch + 1)

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.counter = 0
                self._save_checkpoint()
                logger.info(f"Best model saved (val_acc={val_acc:.4f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        logger.info(f"Training complete. Best val_acc: {self.best_val_acc:.4f}")
        self._log_confusion_matrix()

    def _save_checkpoint(self):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "class_names":      self.class_names,
                "num_classes":      len(self.class_names),
                "architecture":     self.architecture,
                "mode":             self.mode,
            },
            self.save_path,
        )

    def _log_confusion_matrix(self):
        # best checkpoint로 평가 (last epoch 아님)
        ckpt = torch.load(self.save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        all_preds         = []
        all_labels        = []
        all_confs         = []
        all_top3_correct  = []
        all_sex_gt        = []
        all_form_gt       = []
        all_sex_preds     = []
        all_sex_known_gt  = []
        all_form_preds    = []
        all_form_known_gt = []

        with torch.no_grad():
            for batch in self.dataloaders["val"]:
                inputs = batch[0].to(self.device)
                labels = batch[1]

                if self._uses_aux:
                    sex_labels_b  = batch[2]
                    form_labels_b = batch[3]
                    model_out     = self.model(inputs)
                    sp_out        = model_out[0]

                    if self.mode in ("multi_task", "sex_only"):
                        sex_out  = model_out[1]
                        sex_mask = sex_labels_b >= 0
                        all_sex_gt.extend(sex_labels_b.numpy())
                        if sex_mask.sum() > 0:
                            _, sex_preds_b = torch.max(sex_out, 1)
                            all_sex_preds.extend(sex_preds_b.cpu()[sex_mask].numpy())
                            all_sex_known_gt.extend(sex_labels_b[sex_mask].numpy())

                    if self.mode == "multi_task":
                        form_out  = model_out[2]
                        form_mask = form_labels_b >= 0
                        all_form_gt.extend(form_labels_b.numpy())
                        if form_mask.sum() > 0:
                            _, form_preds_b = torch.max(form_out, 1)
                            all_form_preds.extend(form_preds_b.cpu()[form_mask].numpy())
                            all_form_known_gt.extend(form_labels_b[form_mask].numpy())

                    if self.mode == "form_only":
                        form_out  = model_out[1]
                        form_mask = form_labels_b >= 0
                        all_form_gt.extend(form_labels_b.numpy())
                        if form_mask.sum() > 0:
                            _, form_preds_b = torch.max(form_out, 1)
                            all_form_preds.extend(form_preds_b.cpu()[form_mask].numpy())
                            all_form_known_gt.extend(form_labels_b[form_mask].numpy())
                else:
                    sp_out = self.model(inputs)

                probs = F.softmax(sp_out, dim=1)
                confs, preds = probs.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_confs.extend(confs.cpu().numpy())

                k = min(3, sp_out.size(1))
                top3 = torch.topk(sp_out, k=k, dim=1).indices
                correct = top3.cpu().eq(labels.unsqueeze(1)).any(dim=1)
                all_top3_correct.extend(correct.numpy())

        preds_np  = np.array(all_preds)
        labels_np = np.array(all_labels)
        confs_np  = np.array(all_confs)
        accs_np   = (preds_np == labels_np).astype(float)

        # ── Confusion Matrix ─────────────────────────────────────────────
        cm = confusion_matrix(labels_np, preds_np)
        short_names = [c.split("_")[0] for c in self.class_names]
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (val, best epoch)")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # ── ECE + Reliability Diagram ────────────────────────────────────
        n_bins     = 15
        bin_edges  = np.linspace(0, 1, n_bins + 1)
        ece        = 0.0
        bin_accs_plot  = []
        bin_confs_plot = []
        for i in range(n_bins):
            mask = (confs_np >= bin_edges[i]) & (confs_np < bin_edges[i + 1])
            if mask.sum() > 0:
                b_acc  = accs_np[mask].mean()
                b_conf = confs_np[mask].mean()
                ece   += mask.sum() / len(confs_np) * abs(b_acc - b_conf)
                bin_accs_plot.append(b_acc)
                bin_confs_plot.append(b_conf)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.bar(bin_edges[:-1], [accs_np[(confs_np >= bin_edges[i]) & (confs_np < bin_edges[i+1])].mean()
                                 if ((confs_np >= bin_edges[i]) & (confs_np < bin_edges[i+1])).sum() > 0 else 0
                                 for i in range(n_bins)],
               width=1/n_bins, align="edge", alpha=0.4, color="steelblue", label="Accuracy/bin")
        ax.plot(bin_confs_plot, bin_accs_plot, "ro-", markersize=4, label=f"ECE={ece:.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram (val)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        mlflow.log_figure(fig, "reliability_diagram.png")
        plt.close(fig)

        # ── Classification Report ────────────────────────────────────────
        report = classification_report(
            labels_np, preds_np,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics = {}

        for name in self.class_names:
            if name in report:
                metrics[f"f1_{name}"]        = report[name]["f1-score"]
                metrics[f"precision_{name}"] = report[name]["precision"]
                metrics[f"recall_{name}"]    = report[name]["recall"]

        for avg in ("macro avg", "weighted avg"):
            prefix = "macro" if avg == "macro avg" else "weighted"
            metrics[f"{prefix}_f1"]        = report[avg]["f1-score"]
            metrics[f"{prefix}_precision"] = report[avg]["precision"]
            metrics[f"{prefix}_recall"]    = report[avg]["recall"]

        metrics["top3_acc"]    = float(np.mean(all_top3_correct))
        metrics["cohen_kappa"] = float(cohen_kappa_score(labels_np, preds_np))
        metrics["ece"]         = float(ece)

        # ── Auxiliary head metrics ───────────────────────────────────────
        if self._uses_aux:
            # Sex head (multi_task, sex_only)
            if all_sex_gt:
                sex_gt_np = np.array(all_sex_gt)
                metrics["sex_label_coverage"] = float((sex_gt_np >= 0).sum() / len(sex_gt_np))
            if all_sex_preds:
                metrics["sex_acc"]      = float(accuracy_score(all_sex_known_gt, all_sex_preds))
                metrics["sex_macro_f1"] = float(f1_score(all_sex_known_gt, all_sex_preds,
                                                          average="macro", zero_division=0))

            # Form head (multi_task, form_only)
            if all_form_gt:
                form_gt_np = np.array(all_form_gt)
                metrics["form_label_coverage"] = float((form_gt_np >= 0).sum() / len(form_gt_np))
            if all_form_preds:
                metrics["form_acc"]      = float(accuracy_score(all_form_known_gt, all_form_preds))
                metrics["form_macro_f1"] = float(f1_score(all_form_known_gt, all_form_preds,
                                                           average="macro", zero_division=0))

            # 성별별 종 분류 정확도 + 종별 세부 (sex 라벨이 있는 모드만)
            if all_sex_gt:
                sex_gt_np   = np.array(all_sex_gt)
                male_mask   = sex_gt_np == 0
                female_mask = sex_gt_np == 1
                if male_mask.sum() > 0:
                    metrics["male_species_acc"]   = float(accuracy_score(labels_np[male_mask],
                                                                          preds_np[male_mask]))
                if female_mask.sum() > 0:
                    metrics["female_species_acc"] = float(accuracy_score(labels_np[female_mask],
                                                                          preds_np[female_mask]))
                for sp_idx, sp_name in enumerate(self.class_names):
                    sp_mask = labels_np == sp_idx
                    m_mask  = sp_mask & male_mask
                    f_mask  = sp_mask & female_mask
                    if m_mask.sum() >= 3:
                        metrics[f"male_acc_{sp_name}"]   = float(accuracy_score(
                            labels_np[m_mask], preds_np[m_mask]))
                    if f_mask.sum() >= 3:
                        metrics[f"female_acc_{sp_name}"] = float(accuracy_score(
                            labels_np[f_mask], preds_np[f_mask]))

        mlflow.log_metrics(metrics)

    def _masked_loss(self, outputs, labels):
        """label=-1인 샘플은 loss에서 제외. 가중치 없는 CE 사용."""
        mask = labels >= 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.nn.functional.cross_entropy(outputs[mask], labels[mask])

    def _run_epoch(self, phase: str):
        self.model.train() if phase == "train" else self.model.eval()

        running_loss = 0.0
        corrects     = 0

        with torch.set_grad_enabled(phase == "train"):
            for batch in tqdm(self.dataloaders[phase], desc=phase.capitalize()):
                if self._uses_aux:
                    inputs, sp_labels, sex_labels, form_labels = batch
                    sex_labels  = sex_labels.to(self.device)
                    form_labels = form_labels.to(self.device)
                else:
                    inputs, sp_labels = batch

                inputs    = inputs.to(self.device)
                sp_labels = sp_labels.to(self.device)

                if phase == "train":
                    self.optimizer.zero_grad()

                model_out = self.model(inputs)
                if self.mode == "species_only":
                    sp_out = model_out
                    loss   = self.criterion(sp_out, sp_labels)
                elif self.mode == "sex_only":
                    sp_out, sex_out = model_out
                    loss = (self.criterion(sp_out, sp_labels)
                            + self.lambda_sex * self._masked_loss(sex_out, sex_labels))
                elif self.mode == "form_only":
                    sp_out, form_out = model_out
                    loss = (self.criterion(sp_out, sp_labels)
                            + self.lambda_form * self._masked_loss(form_out, form_labels))
                else:  # multi_task
                    sp_out, sex_out, form_out = model_out
                    loss = (self.criterion(sp_out, sp_labels)
                            + self.lambda_sex  * self._masked_loss(sex_out,  sex_labels)
                            + self.lambda_form * self._masked_loss(form_out, form_labels))
                _, preds = torch.max(sp_out, 1)

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects     += torch.sum(preds == sp_labels)

        n          = len(self.dataloaders[phase].dataset)
        epoch_loss = running_loss / n
        epoch_acc  = corrects.double() / n
        return epoch_loss, epoch_acc.item()
