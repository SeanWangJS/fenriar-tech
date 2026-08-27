import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://fenriar-tech.pages.dev/",
    title: "Fenriar Tech",
    description: "Personal tech blog on LLM, CUDA, Systems & Algorithms.",
    author: "SeanWang",
    profile: "https://github.com/SeanWangJS",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Shanghai",
    dir: "ltr",
    googleAnalyticsId: "G-98H3Q9YVS5",
    googleAdsenseId: "ca-pub-4053098847910397",
  },
  posts: {
    perPage: 4,
    perIndex: 4,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: false,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: false,
    },
    search: "pagefind",
  },
  socials: [
    { name: "github", url: "" },
    { name: "x", url: "" },
    { name: "linkedin", url: "" },
    { name: "mail", url: "" },
  ],
  shareLinks: [
    { name: "whatsapp", url: "" },
    { name: "facebook", url: "" },
    { name: "x", url: "" },
    { name: "telegram", url: "" },
    { name: "pinterest", url: "" },
    { name: "mail", url: "" },
  ],
});