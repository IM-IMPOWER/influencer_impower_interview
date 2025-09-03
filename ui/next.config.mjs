/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://api:8000/api/:path*', // hits FastAPI service in docker network
      },
    ];
  },
};

export default {
  async rewrites() {
    return [{ source: '/api/:path*', destination: 'http://api:8000/api/:path*' }];
  },
};

// export default nextConfig;

// /** @type {import('next').NextConfig} */
// const nextConfig = {
//   async rewrites() {
//     return [{ source: '/api/:path*', destination: 'http://api:8000/api/:path*' }];
//   },
// };
// export default nextConfig;
