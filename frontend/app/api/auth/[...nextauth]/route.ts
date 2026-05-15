import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const handler = NextAuth({
  providers: [
    GoogleProvider({
      clientId:     process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      // 최초 로그인 시 Google ID 토큰으로 백엔드 JWT 발급
      if (account?.id_token) {
        const res = await fetch(`${API_BASE}/api/v1/auth/google`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ id_token: account.id_token }),
        });
        if (res.ok) {
          const data = await res.json();
          token.backendToken = data.token;
          token.userId       = data.user_id;
          token.grade        = data.grade;
          token.specialty    = data.specialty;
          token.score        = data.score;
        }
      }
      return token;
    },
    async session({ session, token }) {
      (session as any).backendToken = token.backendToken;
      (session as any).userId       = token.userId;
      (session as any).grade        = token.grade;
      (session as any).specialty    = token.specialty;
      (session as any).score        = token.score;
      return session;
    },
  },
});

export { handler as GET, handler as POST };
