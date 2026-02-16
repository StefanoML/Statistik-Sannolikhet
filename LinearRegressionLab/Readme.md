# Laboration: Linjär Regression(Svenska):

| Instruktioner: Se till att dessa 3 filer ligger i samma mapp: 1: lin_reg.py  2: analysis.ipynb  3: housing.csv |
| -------------------------------------------------------------------------------------------------------------- |

Det här projektet använder ett eget Python-script och en Jupyter Notebook för att förutsäga huspriser baserat på ett tillhandahållet dataset. Huvudmålet var att se vilka faktorer, som inkomst eller läge, som faktiskt styr huspriserna och visa resultaten i en notebook-fil. Jag har byggt projektet med **NumPy** för matematiken och **SciPy** för de statistiska testerna.

### Koden:

Koden använder NumPy för att utföra den matrisalgebra som krävs för den linjära regressionsmodellen. Jag använde "normalekvationen" för att beräkna koefficienterna för varje variabel. Jag använde också SciPy för att hantera statistiska distributioner. Specifikt användes SciPy för att beräkna p-värden och kritiska värden för T-tester och F-test. Dessa används av modellen för att avgöra vilka variabler som är statistiskt signifikanta och för att beräkna 95-procentiga konfidensintervall (konfidensnivån kan ändras till andra värden i notebook-filen). Biblioteket innehåller inga print-kommandon eller kommentarer för att säkerställa att koden helt fokuserar på beräkningar.

### Notebook-filen:

Notebook-filen designades för att uppfylla kravet på att inte använda print-kommandon eller kommentarer i kodcellerna. Istället för att printa text använde jag f-strings för att kombinera etiketter och värden till enstaka objekt. Genom att placera dessa objekt, listor eller ordlistor (dictionaries) på sista raden i en kodcell visar Jupyter dem som en formaterad output. Detta håller notebook-filen ren samtidigt som resultaten syns tydligt. Jag använde markdown-rubriker ovanför varje cell för att förklara vad som händer i koden istället för att använda interna kodkommentarer.

Jag använde raden `np.set_printoptions(precision=2, suppress=True, linewidth=100)` för att göra vissa outputs mer läsbara, särskilt Pearsons korrelationsmatris som annars skulle vara ganska svår att tyda.

### Datahantering och rensning:

Jag använde **genfromtxt** för att ladda in datan istället för vanlig filhantering i Python eftersom den är gjord för att läsa in CSV-filer direkt till NumPy-arrays. Den hanterar automatiskt rubriker och datatyper, vilket förhindrar manuella fel när man omvandlar text till siffror för matematiken, och sparar mycket tid.

För att hantera saknade värden valde jag att ta bort de raderna helt. Även om man kan fylla i dem med medelvärden, så säkerställer borttagning att modellen bara lär sig från faktiskt registrerad data. Eftersom datasetet är stort skadar det inte modellens prestanda att förlora några rader, och det gör resultaten mer pålitliga. Därför inkluderade jag även en liten statistik som extrafunktion som visar hur mycket data som gick förlorad och vilken.

För den kategoriska variabeln (närhet till havet) omvandlade jag kategorierna till siffror med hjälp av "one-hot encoding". Jag exkluderade en kategori under processen för att undvika ett vanligt matematiskt problem där för många relaterade kolumner gör modellens resultat instabila (dummy variable trap).

### Analys av resultat:

Med ett R² på 0,65 förklarar modellen cirka 65% av variansen i huspriser, vilket lämnar 35% oförklarat av de inkluderade variablerna.

F-testet gav ett p-värde nära noll, vilket betyder att modellen som helhet är statistiskt signifikant åtminstone en prediktor har en verklig effekt på priset.

T-tester visar att **median_income** har den starkaste effekten (t = 116,15), vilket betyder att områden med högre inkomster har signifikant högre huspriser. Detta kan återspegla genuina ekonomiska mönster: rikare områden kan ha bättre faciliteter, skolor och infrastruktur som driver upp priserna. Det kan också återspegla ett cirkulärt förhållande där dyra områden attraherar höginkomsttagare som har råd med dessa priser.

Pearsons korrelationsmatris avslöjade allvarlig multikolinjäritet bland bostadsvariablerna: **total_bedrooms** och **households** korrelerar med 0,98, och **total_rooms** korrelerar över 0,90 med båda. Detta betyder att dessa variabler ger nästan identisk information. I den ursprungliga modellen orsakade detta att **total_rooms** fick en kontraintuitiv negativ koefficient. När vi tog bort de redundanta variablerna (**total_bedrooms** och  **households** ) blev  **total_rooms**  koefficienten positiv som förväntat, vilket demonstrerar hur multikolinjäritet kan göra koefficienter instabila och vilseledande.

De kategoriska variablerna för **ocean_proximity** visar signifikanta platseffekter. **ocean_INLAND** har en stark negativ koefficient (cirka -43 000), vilket betyder att fastigheter på inlandet är betydligt billigare än kustnära.  **ocean_ISLAND** -fastigheter visar en stor positiv effekt, dock med ett mycket brett konfidensintervall på grund av det lilla antalet öobservationer i datasetet.

Konfidensintervallen för de flesta prediktorerna är relativt smala (särskilt för **median_income** och geografiska variabler), vilket indikerar precisa skattningar. Denna precision kommer från den stora urvalsstorleken (n = 20 433), vilket gör att modellen kan upptäcka även små effekter tillförlitligt. Dock garanterar inte smala intervall meningsfull tolkning när multikolinjäritet är närvarande vilket ses med bostadsvariablerna i den ursprungliga modellen.

---

# Linear Regression Lab (English):

| Instructions: Make sure 3 files are in the same folder: 1:  lin_reg.py  2: analysis.ipynb  3: housing.csv |
| --------------------------------------------------------------------------------------------------------- |

This project uses a custom Python script and a Jupyter Notebook to predict house prices based on the data set provided. The main goal was to see which factors, like income or location, actually drive house prices and to show those results in a notebook file. I built it using **NumPy** for the math and **SciPy** for the statistical tests.

### The code:

The code uses NumPy to perform the matrix algebra required for the linear regression model. I used the normal equation to calculate the coefficients for each feature. I also used SciPy to handle the statistical distributions. Specifically, SciPy was used to calculate the p-values and critical values for the T-tests and F-test. Those are used by  the model to determine which features are statistically significant and to calculate the 95 percent confidence intervals(the confidence can be set to different values in the notebook). The library itself contains no print statements or comments to ensure the code remains focused purely on calculations.

### The Notebook:

The notebook was designed to meet the requirement of not using print statements or comments in the code cells. Instead of printing text, I used f-strings to combine labels and values into single objects. By placing these objects, lists, or dictionaries on the last line of a code cell, Jupyter displays them as a formatted output. This keeps the notebook clean while still showing the results clearly. I used markdown titles above each cell to explain what is happening in the code instead of using internal code comments.

I used the line np.set_printoptions(precision=2, suppress=True, linewidth=100) to improve the readability of certain outputs, especially the Pearsons' Correlation which returns a matrix that would be quite hard to read otherwise.

### Data Handling and Cleaning:

I used **genfromtxt** to load the data instead of basic Python file handling because it is designed to process CSV files directly into NumPy arrays. It automatically handles things like headers and data types, which prevents manual errors when converting text into numbers for math, and saves a lot of time not having to do those things manually.

To deal with missing values, I chose to remove those rows. While filling them with averages is an option, removing them ensures the model only learns from actual, recorded data. Since the dataset is large, losing a few rows doesn't hurt the model's performance and keeps the results more reliable. That's why I also included as an extra a small statistic showing how much data would be lost and which.

For the categoric feature ( ocean proximity ), I transformed the categories into numbers using "one hot encoding". I excluded one category during this process to avoid a common mathematical issue where having too many related columns makes the model's results unstable (dummy variable trap).

### Results Analysis:


With an R-squared of 0.65, the model explains about 65% of the variance in house prices, leaving 35% unexplained by the included features.

The F-test gave a p-value near zero, which means the model as a whole is statistically significant at least one predictor has a real effect on price.

T-tests show that **median_income** has the strongest effect (t = 116.15), meaning areas with higher incomes have significantly higher house prices. This could reflect genuine economic patterns: wealthier areas may have better amenities, schools, and infrastructure, which drive up prices. It could also reflect a circular relationship where expensive areas attract high income residents who can afford those prices.

The Pearson correlation matrix revealed severe multicollinearity among housing variables: **total_bedrooms** and **households** correlate at 0.98, and **total_rooms** correlates above 0.90 with both. This means these variables provide nearly identical information. In the original model, this caused **total_rooms** to have a counterintuitive negative coefficient. When we removed the redundant variables (**total_bedrooms** and  **households** ), the **total_rooms** coefficient became positive as expected, demonstrating how multicollinearity can make coefficients unstable and misleading.

The categorical variables for **ocean_proximity** show significant location effects. **ocean_INLAND** has a strong negative coefficient (around -43,000), meaning inland properties are substantially cheaper than coastal ones. **ocean_ISLAND** properties show a large positive effect, though with a very wide confidence interval due to the small number of island observations in the dataset.

The confidence intervals for most predictors are relatively narrow (especially for **median_income** and geographic variables), indicating precise estimates. This precision comes from the large sample size (n = 20,433), which allows the model to detect even small effects reliably. However, narrow intervals don't guarantee meaningful interpretation when multicollinearity is present as seen with the housing variables in the original model.
