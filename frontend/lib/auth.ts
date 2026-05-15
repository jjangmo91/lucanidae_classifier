import { useSession } from "next-auth/react";

export function useAuth() {
  const { data: session, status } = useSession();
  const s = session as any;
  return {
    isLoggedIn:    status === "authenticated",
    isLoading:     status === "loading",
    username:      s?.user?.name  ?? null,
    avatar:        s?.user?.image ?? null,
    userId:        s?.userId      ?? null,
    grade:         s?.grade       ?? null,
    specialty:     s?.specialty   ?? null,
    score:         s?.score       ?? 0,
    backendToken:  s?.backendToken ?? null,
  };
}

export function authHeader(token: string | null): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}
