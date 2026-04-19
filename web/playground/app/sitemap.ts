import type { MetadataRoute } from "next"

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: "https://surge-grid.vercel.app",
      lastModified: new Date(),
      changeFrequency: "hourly",
      priority: 1.0,
    },
  ]
}
