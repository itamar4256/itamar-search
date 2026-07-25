const axios = require("axios");
const cheerio = require("cheerio");
const xml2js = require("xml2js");
const fs = require("fs");

async function main() {
    const sitemap = await axios.get("https://www.itamar.club/sitemap.xml");
    const parsed = await xml2js.parseStringPromise(sitemap.data);

    const excluded = [
        "https://www.itamar.club/about-us",
        "https://www.itamar.club/articles"
    ];

    const urls = parsed.urlset.url
        .(u => u.loc[0])
        .filter(url => !excluded.includes(url));

    const articles = [];

    for (const url of urls) {
        console.log("Fetching:", url);

        try {
            const response = await axios.get(url);
            const $ = cheerio.load(response.data);

            const title = $("title").text().trim();
            const paragraphs = [];

$(".sqs-html-content").each((_, block) => {
    $(block)
        .find("p, h1, h2, h3, h4, li, blockquote")
        .each((__, el) => {
            const text = $(el).text().replace(/\s+/g, " ").trim();

            if (text.length > 20) {
                paragraphs.push(text);
            }
        });
});

articles.push({
    id: articles.length + 1,
    title,
    url,
    paragraphs
});

        } catch (err) {
            console.log("Failed:", url);
        }
    }

    fs.writeFileSync(
        "articles.json",
        JSON.stringify(articles, null, 2)
    );

    console.log(`Done. Indexed ${articles.length} pages.`);
}

main();
