import type { MetadataRoute } from "next"

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: "https://surge-omega-nine.vercel.app",
      lastModified: new Date(),
      changeFrequency: "hourly",
      priority: 1.0,
    },
  ]
}
