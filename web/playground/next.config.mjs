/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Turbopack 16.1.x ships a persistent FS cache that corrupts on some
    // drives (yields "Loading persistence directory failed: invalid digit
    // found in string"). Run in memory-only mode — no disk cache, no bug.
    // Fixed upstream in 16.2+: remove once we upgrade.
    turbopackFileSystemCacheForDev: false,
    turbopackFileSystemCacheForBuild: false,
  },
}

export default nextConfig
