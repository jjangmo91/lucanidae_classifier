/** @type {import('next').NextConfig} */

// 개발: http://localhost:8000 / 프로덕션: docker 서비스명 또는 외부 URL
const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig = {
  output: "standalone",
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: true },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND}/api/:path*`,
      },
      {
        source: "/admin/:path*",
        destination: `${BACKEND}/admin/:path*`,
      },
      {
        source: "/uploads/:path*",
        destination: `${BACKEND}/uploads/:path*`,
      },
    ];
  },
};

export default nextConfig;
