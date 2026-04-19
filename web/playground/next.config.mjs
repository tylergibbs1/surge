/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Turbopack 16.1.x ships a persistent FS cache that corrupts on some
    // drives (yields "Loading persistence directory failed: invalid digit
    // found in string"). Run in memory-only mode — no disk cache, no bug.
    // Fixed upstream in 16.2+: remove once we upgrade.
    turbopackFileSystemCacheForDev: false,
    turbopackFileSystemCacheForBuild: false,
    // Rewrite barrel imports into direct module reads at build time.
    // Without this, `import { X } from "@radix-ui/react-icons"` pulls in
    // the whole ~300-icon entry (adds 200-800 ms cold-start eval and
    // inflates HMR). Recharts' root export is the same story — every
    // chart primitive behind one module. Both safe to turn on.
    optimizePackageImports: ["@radix-ui/react-icons", "recharts"],
  },
}

export default nextConfig
