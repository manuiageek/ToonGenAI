import puppeteer from "puppeteer";
import fs from "fs";
import path from "path";
import fetch from "node-fetch";

// CONFIG
const BASE_URL = "https://myanimelist.net/anime/40128/Arte";
const OUTPUT_ROOT_DIR = String.raw`T:\_SELECT\READY\ARTE`;
const OUTPUT_DIR = path.join(OUTPUT_ROOT_DIR, "_CHARACTERS");

// utils
async function downloadImage(url, filePath) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Erreur lors du téléchargement ${url} - ${res.statusText}`);
  }
  const fileStream = fs.createWriteStream(filePath);
  return new Promise((resolve, reject) => {
    res.body.pipe(fileStream);
    res.body.on("error", (err) => reject(err));
    fileStream.on("finish", () => resolve());
  });
}

// main
(async () => {
  let url = BASE_URL.trim();
  if (!url) {
    console.error("Aucune URL définie. Fin du script.");
    process.exit(1);
  }

  if (!url.match(/\/characters\/?$/)) {
    url = url.replace(/\/+$/, "");
    url += "/characters";
  }

  if (!fs.existsSync(OUTPUT_DIR)) {
    console.error(`Le dossier de sortie n'existe pas: ${OUTPUT_DIR}`);
    process.exit(1);
  }
  console.log(`Dossier de sortie: ${OUTPUT_DIR}`);

  const browser = await puppeteer.launch({
    headless: true,
    ignoreHTTPSErrors: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();

  await page.setUserAgent(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
  );

  try {
    await page.goto(url, { waitUntil: "networkidle2" });
    await page.waitForSelector("h3.h3_character_name", { timeout: 10000 });
    console.log("La page des personnages est chargée, début de l'extraction des liens...");

    const characterLinks = await page.evaluate(() => {
      const elements = Array.from(document.querySelectorAll("h3.h3_character_name"));
      return elements
        .map((el) => (el.parentElement ? el.parentElement.href : null))
        .filter((link) => link !== null);
    });
    console.log(`Nombre de personnages trouvés : ${characterLinks.length}`);

    const results = [];
    const totalCharacters = characterLinks.length;

    for (let i = 0; i < totalCharacters; i++) {
      const link = characterLinks[i];
      console.log(`Traitement du personnage ${i + 1}/${totalCharacters} : ${link}`);

      const urlName = link.split("/").pop();
      const name = (urlName || "unknown").replace(/[^a-zA-Z0-9-_]/g, "");

      await page.goto(link, { waitUntil: "networkidle2" });

      let imageAvailable = true;
      try {
        await page.waitForSelector("img.portrait-225x350", { timeout: 30000 });
      } catch {
        console.log(`Image non disponible pour ${name}.`);
        imageAvailable = false;
      }

      let imageUrl = "";
      if (imageAvailable) {
        const details = await page.evaluate(() => {
          const imgElement = document.querySelector("img.portrait-225x350");
          const imageUrl =
            imgElement?.getAttribute("data-src") ||
            imgElement?.getAttribute("src") ||
            "";
          return { imageUrl };
        });
        imageUrl = details.imageUrl;
        console.log("Nom extrait :", name);
        console.log("URL de l'image :", imageUrl);

        if (imageUrl) {
          const noQuery = imageUrl.split("?")[0];
          const ext = path.extname(noQuery) || ".jpg";
          const fileName = `${name}${ext}`;
          const filePath = path.join(OUTPUT_DIR, fileName);

          if (fs.existsSync(filePath)) {
            console.log(`Déjà présent, on saute: ${filePath}`);
          } else {
            try {
              console.log(`Téléchargement de l'image vers ${filePath}`);
              await downloadImage(imageUrl, filePath);
              console.log("Téléchargement réussi !");
            } catch (downloadError) {
              console.error("Erreur lors du téléchargement de l'image :", downloadError);
            }
          }
        } else {
          console.log(`Aucune URL d'image trouvée pour ${name}.`);
        }
      } else {
        console.log(`Aucune image à télécharger pour ${name}.`);
      }

      results.push({ url: link, name, imageUrl: imageUrl || null });

      await page.goto(url, { waitUntil: "networkidle2" });
      await page.waitForSelector("h3.h3_character_name", { timeout: 10000 });
    }

    console.log("Tous les personnages extraits :", results);
  } catch (error) {
    console.error("Erreur lors du scraping :", error);
  } finally {
    await browser.close();
  }
})();