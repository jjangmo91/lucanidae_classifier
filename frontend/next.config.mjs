/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        // 프론트에서 /api/v1/* 요청 → FastAPI localhost:8000 으로 전달
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/admin/:path*",
        destination: "http://localhost:8000/admin/:path*",
      },
      {
        source: "/uploads/:path*",
        destination: "http://localhost:8000/uploads/:path*",
      },
    ];
  },
};

export default nextConfig;
