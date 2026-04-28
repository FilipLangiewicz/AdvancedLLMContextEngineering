# NLP - dokumentacja projektu
## Context Engineering - zaawansowane zarządzanie kontekstem

Implementacja i ewaluacja technik zaawansowanego zarządzania kontekstem (context engineering) w systemach agentowych/RAG.

### Zadania:
- Zbadać i zaimplementować system konwersacyjny (np. czat) obsługujący zarządzanie długim kontekstem (np. kilkunasto/kilkudziesięcio stronicowe dokumenty, >10 tys. tokenów per dokument) z wybranymi technikami np. semantic chunking, document structure-based chunking w wybranej dziedzinie (np. inżynieria, IT, programowanie itp.)
- Zaimplementować wybraną metodę query understanding np. query rewriter
- Porównać co najmniej 2 strategie context engineering np. context compression/summarizing, hierarchical summarization
- Zbadanie oraz zaimplementowanie wybranej metody context cache/semantic cache
- Zbadanie problemu "Lost-in-the-middle" i sposobów jego mitygacji

## Zespół projektowy:
- Natalia Choszczyk
- Filip Langiewicz
- Mikołaj Rowicki

## Definicja problemu

Kluczowym wyzwaniem współczesnych systemów RAG jest efektywne zarządzanie długim kontekstem, modele językowe dysponują skończonym oknem kontekstowym i nie są w stanie przetworzyć obszernych dokumentów w całości. Projekt skupia się na domenie prawnej, gdzie akty, kodeksy i regulaminy liczą dziesiątki lub setki stron, a precyzja i kompletność odpowiedzi ma szczególne znaczenie. W tym kontekście identyfikujemy następujące problemy:

- **Zarządzanie długim kontekstem:** dokumenty prawne przekraczające 10 000 tokenów wymagają podziału na fragmenty. Standardowy podział stałej długości nie uwzględnia struktury prawnej dokumentu, co prowadzi do rozbicia logicznie powiązanych przepisów i utraty kontekstu semantycznego.
- **Rozumienie zapytań:** pytania użytkowników są często sformułowane w języku potocznym, niedostosowanym do struktury bazy wiedzy, co bez mechanizmu ich przekształcania obniża trafność wyszukiwania.
- **Porównanie strategii zarządzania kontekstem:** brakuje empirycznych podstaw do oceny, która strategia przetwarzania kontekstu, kompresja, hierarchiczne podsumowywanie czy inne, daje najlepsze wyniki w domenie prawnej.
- **Efektywność przetwarzania:** brak mechanizmu cache'owania powoduje, że każde zapytanie dotyczące tych samych dokumentów generuje zbędne obciążenie systemu.
- **Zjawisko „lost-in-the-middle":** modele językowe mają tendencję do pomijania informacji ze środkowych części kontekstu, co w dokumentach prawnych może skutkować pominięciem kluczowych sekcji.

## Przegląd literatury

### RAG i zarządzanie kontekstem

Retrieval-Augmented Generation zostało wprowadzone przez Patrick Lewis i in. (2020) jako podejście łączące modele językowe z zewnętrznym wyszukiwaniem dokumentów. Autorzy pokazują, że integracja dokumentów źródłowych w procesie generacji poprawia trafność odpowiedzi i ogranicza halucynacje, szczególnie w zadaniach wymagających wiedzy opartej na danych [1]. W kontekście prawa RAG jest szczególnie istotny ze względu na dużą objętość i zmienność aktów prawnych.

### Chunking i reprezentacja dokumentów

Literatura wskazuje, że proste dzielenie tekstu na fragmenty o stałej długości jest niewystarczające dla złożonych dokumentów. Reuter et al. (2025) proponują Summary-Augmented Chunking, w którym każdy fragment jest wzbogacony o globalne streszczenie dokumentu, co poprawia trafność wyszukiwania [2]. Równolegle rozwijane są podejścia semantyczne (dzielenie według treści) i strukturalne (np. według artykułów i paragrafów), które lepiej zachowują spójność logiczną.

### Query rewriting

Xinbei Ma i in. (2023) proponują schemat Rewrite-Retrieve-Read, w którym zapytanie użytkownika jest najpierw reformułowane do postaci bardziej zgodnej z językiem dokumentów i wymaganiami wyszukiwarki [3]. Zeqiu Wu i in. (2022) w pracy nad CONQRR pokazują, że przepisywanie zapytań konwersacyjnych do formy niezależnej (stand-alone) poprawia skuteczność wyszukiwania, zwłaszcza w systemach typu czat [4].

### Kompresja kontekstu i podsumowywanie

Shuyu Guo i in. (2025) wprowadzają ACC-RAG, który adaptacyjnie kompresuje kontekst poprzez adaptacyjną kompresję z hierarchicznym wyborem istotnych fragmentów, co pozwala zmniejszyć koszty obliczeniowe bez istotnej utraty jakości [5]. Xin Cheng i in. (2024) proponują xRAG, gdzie dokument reprezentowany jest jako pojedynczy embedding-token, co znacząco redukuje długość wejścia [6]. Z kolei Litu Ou i Mirella Lapata (2025) analizują hierarchiczne podsumowywanie, wskazując, że brak dostępu do oryginalnego kontekstu podczas agregacji zwiększa ryzyko generowania nieścisłości [7].

### Cache w systemach RAG

Chao Jin i in. (2024) proponują RAGCache, mechanizm cache'owania fragmentów wiedzy i stanów modelu w architekturze hierarchicznej. Praca pokazuje, że takie podejście znacząco redukuje czas do wygenerowania pierwszego tokenu (TTFT) i zwiększa przepustowość [8].

### Problem „lost-in-the-middle"

Nelson F. Liu i in. (2023) wykazują, że modele językowe mają trudności z wykorzystaniem informacji znajdujących się w środkowej części długiego kontekstu [9]. Gongbo Zhang i in. (2025) proponują BriefContext, podejście dzielące kontekst na części i agregujące wyniki (map-reduce), co pozwala lepiej wykorzystać rozproszone informacje i poprawia jakość odpowiedzi [10].


## Opis rozwiązania
Projekt obejmuje modułowy system RAG dla dziedziny prawa energetycznego. Celem jest porównanie różnych technik zaawansowanego zarządzania kontekstem (context engineering) w pracy z obszernymi dokumentami o formalnym charakterze. Projekt został podzielony na pięć sekcji przedstawionych poniżej.

### 1. Dane
Dokumenty PDF wykorzystane do realizacji projektu zostały umieszczone w katalogu `data/pdf` (`document1.pdf`, `document2.pdf`, `document3.pdf`). Zawierają one akty i regulacje prawne dotyczące sektora energetycznego, a w szczególności:

- **Dokument 1:** Ustawa z dnia 20 maja 2016 r. o efektywności energetycznej (33 strony),
- **Dokument 2:** Rozporządzenie Ministra Klimatu z dnia 7 kwietnia 2020 r. w sprawie szczegółowych zasad kształtowania i kalkulacji taryf oraz rozliczeń z tytułu zaopatrzenia w ciepło (23 strony),
- **Dokument 3:** Ustawa z dnia 10 kwietnia 1997 r., Prawo energetyczne (428 stron). 

### 2. Przetwarzanie dokumentów i baza wektorowa
Zaimplementowany został pipeline przetwarzania dokumentów obejmujący wczytywanie plików PDF, czyszczenie treści (usuwanie nagłówków Kancelarii Sejmu, datowników, zbędnych białych znaków oraz łączenie wyrazów rozdzielonych dywizami przy końcu wiersza), podział tekstu na fragmenty oraz zapis reprezentacji semantycznych w bazie wektorowej Qdrant. Dla każdego dokumentu utrzymywane są dwie kolekcje, odpowiadające dwóm strategiom chunkingu, dzięki czemu możliwe jest porównanie wpływu segmentacji na jakość odpowiedzi systemu.

Strategia strukturalna wykorzystuje cechy charakterystyczne polskich aktów prawnych. Tekst dzielony jest na fragmenty wyznaczane przez nagłówki artykułów (`Art. N.`) oraz paragrafów (`§ N.`), z zachowaniem informacji o nadrzędnym rozdziale. Każdy fragment otrzymuje metadane obejmujące numer artykułu, rozdział, źródłowy plik PDF oraz numer strony, co umożliwia precyzyjne odwoływanie się do oryginalnego dokumentu w odpowiedzi systemu. Fragmenty znacznie przekraczające docelowy rozmiar są dzielone rekurencyjnie z zachowaniem kontekstu artykułu, a w przypadku dokumentów bez rozpoznawalnej struktury aktywuje się mechanizm awaryjny oparty na rekurencyjnym splicie tekstowym.

Strategia semantyczna wykorzystuje `SemanticChunker` z biblioteki LangChain Experimental, który dzieli tekst w miejscach istotnej zmiany znaczenia mierzonej dystansem między embeddingami sąsiednich zdań. Dokumenty są wcześniej scalane do pełnego tekstu w obrębie źródła, a zbyt krótkie fragmenty są dołączane do poprzedzających, aby uniknąć tworzenia chunków pozbawionych wystarczającego kontekstu.

Embeddingi obliczane są modelem `paraphrase-multilingual-mpnet-base-v2`, dobranym ze względu na dobre wsparcie języka polskiego oraz dostępność lokalnie. Wektory zapisywane są w Qdrant z metryką podobieństwa cosinusowego.

### 3. Pipeline RAG
Pipeline RAG dotyczy etapów prowadzących od analizy zapytania (promptu) do wygenerowania odpowiedzi na podstawie kontekstu z bazy wektorowej. Architektura jest modułowa, każdy z komponentów można włączać, wyłączać lub wymieniać niezależnie. Główne etapy przetwarzania obejmują:

- **query understanding**, doprecyzowanie treści zapytania. Zaimplementowana została strategia rewriting, która z pomocą LLM przeredagowuje zapytanie użytkownika na bardziej formalną wersję odpowiadającą stylowi dokumentów prawnych. Komponent zwraca listę przepisanych zapytań, co pozwala w przyszłości łatwo dołączyć inne strategie (multi-query, HyDE, step-back),
- **retrieval**, pobranie trafnych fragmentów kontekstu z bazy wektorowej za pomocą wyszukiwania semantycznego. Liczba zwracanych fragmentów jest parametryzowalna (domyślnie k=5),
- **reranking**, uporządkowanie wyników wyszukiwania. Dostępne są dwie opcje: jedna zachowująca domyślną kolejność fragmentów oraz druga korzystająca z metody U-shaped, która stanowi jednocześnie metodę mitygacji zjawiska "lost-in-the-middle". W metodzie U-shape najistotniejsze fragmenty umieszczane są na początku i na końcu listy, mniej istotne w środku,
- **context compression**, skrócenie kontekstu do kluczowych treści. Zaimplementowane zostały trzy metody. Filtr ekstrakcyjny oparty o `LLMChainExtractor` z LangChain analizuje każdy fragment osobno i usuwa nieistotne pasaże. Metoda hierarchiczna w pierwszym kroku streszcza każdy chunk w kontekście pytania, a następnie agreguje streszczenia w jedno spójne podsumowanie. Metoda BriefContext (inspirowana pracą Zhang et al. 2025) dzieli pobrane fragmenty na trzy partycje, generuje odpowiedź dla każdej z nich osobno, a następnie agreguje uzyskane odpowiedzi w jedną finalną. BriefContext stanowi alternatywną metodę mitygacji zjawiska "lost-in-the-middle", ponieważ żadna z partycji przekazywanych do modelu nie jest na tyle długa, aby informacja zginęła w środku,
- **generation**, moduł, który buduje finalny kontekst oraz odpytuje model LLM na podstawie zdefiniowanego promptu systemowego. Prompt wymusza odpowiedź ściśle bazującą na przesłanym kontekście oraz odmowę odpowiedzi w przypadku braku odpowiednich informacji. Architektura wspiera czterech dostawców LLM (Groq, Google, OpenAI, Anthropic) wybieranych przez konfigurację.

Analogiczne zdefiniowanie kolejnych kroków i metod prowadzących do wygenerowania najlepszej odpowiedzi zostało przedstawione w schemacie "Rewrite-Retrieve-Read" [3].

Dodatkowo zastosowana została metoda semantic cache, która optymalizuje obsługę powtarzalnych zapytań zbliżonych semantycznie. Dla nowego zapytania wyliczany jest embedding i porównywany z embeddingami wcześniejszych zapytań za pomocą metryki cosine similarity. Jeśli inne zapytanie jest wystarczająco podobne, na podstawie ustalonego progu (domyślnie 0.92), zwracana jest wcześniej wygenerowana odpowiedź. Cache działa po stronie aplikacji i jest niezależny od mechanizmu KV-cache w samym modelu.

### 4. Aplikacja Streamlit
Przygotowana została aplikacja w formie czatu z wykorzystaniem pakietu Streamlit. Interfejs umożliwia zmianę najważniejszych ustawień systemu w czasie rzeczywistym: strategii chunkingu, strategii rerankingu, metody kompresji, włączenia query rewritingu, dostawcy modelu, liczby pobieranych fragmentów oraz progu cache. Po przesłaniu zapytania aplikacja zwraca odpowiedź wraz z dokładnym odwołaniem do źródła (numer dokumentu, strona, artykuł, rozdział) oraz informacją, czy odpowiedź pochodzi z pamięci podręcznej.

### 5. Plan eksperymentów i ewaluacji
Ewaluacja systemu RAG w domenie prawnej została przeprowadzona w dwóch komplementarnych osiach. Pierwsza dotyczy walidacji kontekstu, oceny jakości fragmentów dokumentów przekazywanych do modelu generatywnego, czyli tego czy retriever odnajduje właściwe fragmenty oraz czy kolejne etapy przetwarzania kontekstu (reranking, kompresja) zachowują istotne informacje. Druga oś dotyczy jakości odpowiedzi, czyli oceny tekstu wygenerowanego przez model na podstawie tego kontekstu. Rozdzielenie tych dwóch osi jest istotne, ponieważ pozwala zlokalizować źródło ewentualnych błędów: słaba odpowiedź może wynikać zarówno z niedoskonałego retrievala, jak i z ograniczeń samego modelu generatywnego.

#### Metryki

Walidacja kontekstu została przeprowadzona z wykorzystaniem biblioteki Ragas, oferującej referencyjne implementacje miar oceny systemów RAG. Wykorzystano metryki *context precision* (czy w dostarczonych fragmentach faktycznie znajdują się informacje istotne dla pytania) oraz *context relevance* (jaki odsetek treści fragmentów odnosi się bezpośrednio do zapytania). Obie metryki nie wymagają złotych odpowiedzi, operują wyłącznie na trójce pytanie, kontekst, odpowiedź, co czyni je odpowiednimi dla naszego scenariusza bez ground truth.

Jakość odpowiedzi została oceniona własną implementacją LLM-as-a-Judge opartą na większym modelu językowym niż wykorzystywany w głównym pipeline'ie generacyjnym. Wybór własnej implementacji zamiast gotowych metryk Ragas wynika z konieczności dostosowania promptów do języka polskiego i specyfiki tekstów prawnych. Mierzone są trzy metryki w skali 1-5 wraz z uzasadnieniem przedstawionym przez model oceniający:

- *faithfulness*, czy każde stwierdzenie w odpowiedzi wynika z dostarczonego kontekstu. Metryka kluczowa dla domeny prawnej, gdzie halucynacje są niedopuszczalne,
- *answer relevance*, czy odpowiedź faktycznie odnosi się do zadanego pytania, a nie omija jego sedna,
- *completeness*, czy odpowiedź zawiera konkretną, merytoryczną treść. Metryka rozróżnia odpowiedzi pełne od tych powierzchownych lub błędnie odmownych. Dla pytań, na które odpowiedź nie znajduje się w kontekście, poprawna odmowa oceniana jest wysoko, natomiast halucynacja niskie.

Metryka completeness została dodana podczas pierwszych eksperymentów, ponieważ same faithfulness i answer relevance nie różnicowały wystarczająco strategii. Wczesne wyniki pokazywały, że odpowiedzi typu "nie znalazłem informacji" otrzymywały maksymalne oceny faithfulness i relevance (ponieważ odmowa technicznie pasuje do kontekstu i adresuje pytanie), co maskowało istotne różnice w jakości retrievala między konfiguracjami. Completeness wprowadza wymiar, w którym odmowa odpowiedzi mimo dostępnej informacji w kontekście ocenia się nisko.

Uzupełnieniem powyższych są metryki ilościowe: czas odpowiedzi oraz liczba tokenów wejściowych i wyjściowych, ze szczególnym uwzględnieniem efektywności strategii kompresji mierzonej stosunkiem liczby tokenów po kompresji do liczby tokenów przed nią.

#### Zestaw testowy

Zestaw testowy został przygotowany ręcznie na podstawie wiedzy eksperckiej z dziedziny prawa energetycznego. Pytania zostały podzielone na trzy kategorie. Pytania z odpowiedzią krótką (S1, S2, S3, S4) obejmują definicje, zasady i konkretne wartości pozwalające na precyzyjną weryfikację. Pytania z odpowiedzią rozbudowaną (L1, L2, L3) dotyczą procedur i kompleksowych wyjaśnień, testując kompletność i strukturę odpowiedzi. Pytania kontrolne (N1, N2, N3) to pytania, na które dokumenty *nie* zawierają odpowiedzi, ich celem jest weryfikacja odporności systemu na halucynacje i sprawdzenie, czy model rzeczywiście odmawia odpowiedzi zamiast ją zmyślać.

#### Eksperymenty

Pierwszym eksperymentem było porównanie strategii context engineering. Każda z testowanych konfiguracji systemu została uruchomiona dla wszystkich pytań z zestawu testowego, a wyniki zebrano w macierz pozwalającą ocenić wpływ poszczególnych komponentów na jakość odpowiedzi i koszt obliczeniowy. Porównane konfiguracje obejmowały: baseline (chunking strukturalny, brak rerankingu, brak kompresji), chunking semantyczny, reranking U-shape, query rewriting, kompresję ekstrakcyjną, kompresję hierarchiczną, BriefContext oraz konfigurację łączącą rewriting + U-shape + kompresję hierarchiczną.

Drugim eksperymentem było kontrolowane badanie zjawiska lost-in-the-middle. Polegało ono na ręcznym ułożeniu kontekstu w taki sposób, by fragment zawierający odpowiedź na pytanie znalazł się na różnych pozycjach (1, 3, 5, 7 i 10 z dziesięciu pobranych fragmentów), a następnie zmierzeniu wpływu tej pozycji na faithfulness, answer relevance i completeness. Eksperyment został powtórzony dla trzech metod mitygacji: brak mitygacji (kolejność oryginalna), reranking U-shape oraz kompresja BriefContext. Wybór kluczowego fragmentu dla każdego pytania został dokonany automatycznie przez większy model językowy (Gemini 2.5 Flash), co zapewniło powtarzalność i obiektywizm selekcji. Zestaw testowy LITM został ograniczony do pytań, dla których odpowiedź faktycznie znajduje się w dokumentach (S1, S2, L3).

#### Konfiguracja techniczna ewaluacji

Modelem generatywnym wykorzystywanym w pipeline'ie był Llama 3.3 70B uruchomiony przez API Groq, ze względu na dostępność w darmowym tieru oraz dobre radzenie sobie z polskim. Sędzią LLM-as-a-Judge był Gemini 2.5 Flash (a w eksperymencie LITM Gemini 2.5 Flash-Lite ze względu na większą stabilność dziennych limitów), świadomie wybrany jako model spoza rodziny Llama, aby uniknąć biasu samooceny. Wszystkie eksperymenty były idempotentne, w przypadku przerwania (np. wyczerpania dziennego limitu tokenów Groq) skrypty kontynuowały od miejsca zatrzymania bez utraty postępu.

## Wyniki eksperymentów

### Macierz konfiguracji

Tabela poniżej zawiera średnie wartości metryk dla każdej z ośmiu konfiguracji, uzyskane na zestawie 10 pytań testowych (4 krótkie, 3 długie, 3 negatywne).

| Konfiguracja      | Faithfulness | Relevance | Completeness | Latencja [s] | Kontekst [tok.] | Odpowiedź [tok.] |
|-------------------|--------------|-----------|--------------|--------------|-----------------|------------------|
| baseline          | 5.00         | 5.00      | 4.60         | 0.72         | 698             | 21               |
| semantic_chunk    | 5.00         | 5.00      | 4.20         | 7.17         | 1633            | 17               |
| u_shape           | 5.00         | 4.80      | 4.80         | 0.70         | 698             | 21               |
| rewrite           | 4.80         | 5.00      | 4.20         | 1.24         | 721             | 44               |
| extractive        | 5.00         | 5.00      | 5.00         | 2.14         | 106             | 35               |
| hierarchical      | 5.00         | 4.80      | 4.40         | 3.63         | 98              | 54               |
| full_stack        | 5.00         | 4.60      | 4.50         | 4.01         | 92              | 54               |
| brief_context     | 5.00         | 4.80      | 4.80         | 2.70         | 26              | 10               |

### Obserwacje z macierzy

**Brak halucynacji.** Wszystkie konfiguracje uzyskały faithfulness na poziomie 4.80-5.00, co potwierdza skuteczność promptu systemowego wymuszającego odpowiedzi oparte wyłącznie na dostarczonym kontekście. System nie produkuje informacji, których nie ma w dokumentach, niezależnie od konfiguracji. To istotny wynik dla domeny prawnej, w której halucynacje są szczególnie niebezpieczne.

**Chunking strukturalny vs semantyczny.** Strategia strukturalna (baseline) okazała się skuteczniejsza i znacząco tańsza od semantycznej dla naszego zestawu pytań i dokumentów. Baseline uzyskał wyższą completeness (4.60 vs 4.20) przy 2.3 razy mniejszym kontekście (698 vs 1633 tokenów) i 10 razy krótszej latencji (0.72s vs 7.17s). Wynik ten potwierdza tezę, że dla dokumentów o silnej strukturze formalnej (artykuły, paragrafy) podział oparty na tej strukturze lepiej zachowuje spójność logiczną fragmentów niż podział oparty na podobieństwie semantycznym.

**U-shape reranking.** Uzyskał najwyższą wartość completeness wśród konfiguracji bez kompresji (4.80) przy nieznacznie niższej answer relevance (4.80). Zmiana kolejności fragmentów wpływa na ocenę odpowiedzi przy tym samym retrievalu, ale w stosunkowo niewielkim stopniu, co sugeruje, że na naszym zestawie pytań efekt lost-in-the-middle dla k=5 jest umiarkowany. Bardziej szczegółowe badanie tego zjawiska zostało przeprowadzone w eksperymencie pozycyjnym.

**Query rewriting.** Wbrew początkowym oczekiwaniom rewriting zaszkodził completeness (4.20 vs 4.60 baseline). Analiza odpowiedzi pokazuje, że rewriting czasem zmienia akcent zapytania (przykład: zapytanie o ogólną listę kosztów zostało przeformułowane na konkretny aspekt obowiązku z ustawy o efektywności energetycznej, co spowodowało retrieval innych fragmentów). Dla zestawu pytań już sformułowanych formalnym językiem prawnym rewriting może być niepotrzebny lub wręcz szkodliwy.

**Kompresja: ogromna oszczędność tokenów przy zachowaniu jakości.** Hierarchical zmniejsza kontekst z 698 do 98 tokenów (7-krotna redukcja), tracąc tylko 0.20 punktu na completeness (4.40). BriefContext idzie dalej, redukuje kontekst do 26 tokenów (27-krotna redukcja względem baseline) przy jednocześnie wyższej completeness (4.80). Extractive pokazuje pozornie najwyższy wynik (5.00), ale wymaga ostrożnej interpretacji: w 6 z 10 przypadków filtr ekstrakcyjny zwrócił pusty kontekst, co wymusiło odmowę odpowiedzi przez model. Dla pytań negatywnych odmowa jest zachowaniem prawidłowym i otrzymuje wysoką ocenę completeness, co podbija średnią. Tym samym extractive jest skuteczny tam, gdzie kontekst i tak nie zawiera odpowiedzi, ale traci informacje także w przypadkach, gdy odpowiedź jest dostępna.

**BriefContext jako mityacja LITM.** Konfiguracja BriefContext osiągnęła całkowitą jakość porównywalną z najlepszymi konfiguracjami (completeness 4.80, faithfulness 5.00) przy najmniejszym kontekście wszystkich konfiguracji (26 tokenów). Wynik ten potwierdza, że podejście map-reduce skutecznie eliminuje problem długiego kontekstu. Każda z trzech partycji generuje odpowiedź na podstawie krótszego, lokalnego fragmentu, a finalna agregacja łączy wszystkie istotne informacje. Koszt to wzrost latencji do 2.70s (głównie czas generacji odpowiedzi cząstkowych).

**Full stack.** Konfiguracja łącząca rewriting, U-shape i kompresję hierarchiczną nie poprawiła wyników względem prostszych konfiguracji (completeness 4.50). Sugeruje to, że techniki context engineering nie składają się addytywnie, niektóre z nich mogą się kanibalizować. Rewriting zmienia retrieval, a kompresja hierarchiczna gubi wówczas inne informacje niż w przypadku samego rewritingu.

### Wyniki Ragas

| Konfiguracja      | Context Precision | Context Relevance | N  |
|-------------------|-------------------|-------------------|----|
| baseline          | 0.175             | 0.450             | 10 |
| semantic_chunk    | 0.120             | 0.425             | 10 |
| u_shape           | 0.170             | 0.450             | 10 |
| rewrite           | 0.162             | 0.450             | 10 |
| extractive        | 0.333             | 0.875             | 4  |
| hierarchical      | 1.000             | 0.550             | 10 |
| full_stack        | 1.000             | 0.700             | 10 |
| brief_context     | 1.000             | 0.200             | 10 |

Wyniki Ragas wymagają szczególnej ostrożności w interpretacji ze względu na sposób, w jaki Ragas traktuje wynik kompresji. Dla konfiguracji bez kompresji (`baseline`, `semantic_chunk`, `u_shape`, `rewrite`) Ragas otrzymuje listę 5 oryginalnych fragmentów z bazy wektorowej i ocenia każdy z osobna. W tym przypadku metryki działają zgodnie z zamierzonym sensem: niska precision (0.12-0.18) oznacza, że spośród 5 zwróconych fragmentów tylko mniej więcej jeden jest faktycznie istotny dla wygenerowania odpowiedzi.

Dla konfiguracji z kompresją (`hierarchical`, `brief_context`, `full_stack`) Ragas otrzymuje już *wynik kompresji*, czyli pojedynczy zsyntetyzowany fragment. Z definicji ten jeden fragment jest istotny (wszak posłużył do wygenerowania odpowiedzi), więc precision automatycznie wynosi 1.00. Liczba ta nie odzwierciedla jakości oryginalnego retrievala, lecz jedynie fakt, że zsyntetyzowany kontekst został wykorzystany. Z tego względu precision dla konfiguracji z kompresją należy interpretować ostrożnie i raczej w odniesieniu do relevance.

Context relevance dla konfiguracji z kompresją jest bardziej miarodajne. Najwyższe wyniki uzyskał `full_stack` (0.700) oraz `hierarchical` (0.550), co wskazuje, że ich zsyntetyzowane streszczenia faktycznie odnoszą się do zadanego pytania. BriefContext uzyskał niski wynik (0.200) z powodu specyficznego zachowania: dla 8 z 10 pytań agregacja zakończyła się stwierdzeniem braku informacji w dokumentach, co Ragas ocenia jako nieistotne dla zapytania, mimo że jest to merytorycznie poprawne zachowanie. Dla pytań pozytywnych S1 i S2 BriefContext uzyskał maksymalne relevance (1.00). Wniosek jest taki, że Ragas relevance penalizuje odmowy odpowiedzi nawet w przypadkach, gdy są one merytorycznie uzasadnione.

Dla konfiguracji `extractive` Ragas zwrócił wyniki tylko dla 4 z 10 pytań, ponieważ w pozostałych przypadkach kontekst po ekstrakcji był pusty i nie nadawał się do oceny. Precision 0.333 i relevance 0.875 dla tych 4 pytań sugeruje, że tam gdzie filtr ekstrakcyjny coś zostawia, jest to wysoce istotne, ale nie zachowuje wszystkich istotnych fragmentów oryginalnego retrievala.

Wyniki dla konfiguracji bez kompresji są spójne z głównymi wynikami eksperymentu. Niewielkie różnice precision i relevance między baseline (0.175 / 0.450), u_shape (0.170 / 0.450) i rewrite (0.162 / 0.450) potwierdzają, że jakość samego retrievala jest podobna we wszystkich tych konfiguracjach, różnice w completeness wynikają głównie z kolejności podawania fragmentów i sformułowania zapytania. Semantic chunk uzyskał najniższy wynik precision (0.120) mimo największego kontekstu, co potwierdza obserwację z głównego eksperymentu, że więcej kontekstu nie oznacza lepszego retrievala.

### Eksperyment lost-in-the-middle

Eksperyment polegał na zmuszeniu pipeline'u do wygenerowania odpowiedzi przy kontekście z kluczowym fragmentem ułożonym w określonym miejscu listy 10 pobranych fragmentów. Sprawdzono trzy metody mitygacji: brak mitygacji (kolejność po retrievalu), reranking U-shape oraz kompresję BriefContext. Tabela poniżej zawiera średnią completeness uśrednioną po trzech pytaniach (S1, S2, L3) dla każdej kombinacji metody i pozycji.

| Mitigation     | pos=1 | pos=3 | pos=5 | pos=7 | pos=10 | średnia |
|----------------|-------|-------|-------|-------|--------|---------|
| brak           | 3.00  | 3.00  | 3.00  | 3.00  | 4.33   | 3.27    |
| u_shape        | 3.00  | 3.67  | 3.67  | 4.33  | 3.67   | 3.67    |
| brief_context  | 3.67  | 4.33  | 4.33  | 4.00  | 3.00   | 3.87    |

Wyniki potwierdzają obecność efektu lost-in-the-middle w naszym systemie, choć w nieoczywistej postaci. Bez mitygacji jakość odpowiedzi jest jednostajnie niska na pozycjach 1-7 (3.00) i wzrasta dopiero na pozycji 10 (4.33), co wskazuje raczej na efekt "primacy/recency" ze szczególnym faworyzowaniem końca kontekstu. Reranking U-shape lekko wyrównuje krzywą, najwyższy wynik osiąga na pozycji 7 (4.33), co jest spójne z założeniem metody, że środek listy w wersji oryginalnej zostaje przesunięty na koniec po reorderingu.

BriefContext skutecznie eliminuje wpływ pozycji w środkowych obszarach kontekstu, najwyższe wyniki osiąga na pozycjach 3 i 5 (4.33), gdzie metoda bez mitygacji najgorzej sobie radzi. Spadek na pozycji 10 (3.00) jest interesujący, prawdopodobnie wynika z tego, że gdy kluczowy fragment trafia do ostatniej partycji, jego informacja gubi się w fazie agregacji częściowych odpowiedzi.

Średnio BriefContext osiąga najwyższą completeness (3.87), wyprzedzając U-shape (3.67) i konfigurację bez mitygacji (3.27). Wzór mitygacji pozycyjnej różni się jednak między metodami: U-shape pomaga w środkowych pozycjach przez przesunięcie ich na końce, BriefContext eliminuje problem przez podział kontekstu na partycje na tyle krótkie, że żadna informacja nie ginie w środku. Wyniki sugerują, że obie metody są komplementarne, U-shape jest tańsza obliczeniowo i nie wymaga dodatkowego wnioskowania, natomiast BriefContext zachowuje wyższą jakość i działa stabilniej na różnych pozycjach.

Faithfulness w eksperymencie LITM utrzymywała się na wysokim poziomie niezależnie od pozycji i metody (4.47-5.00), co potwierdza, że pipeline nie halucynuje nawet w trudnych konfiguracjach kontekstu. Najniższe wartości dotyczą U-shape (4.47), prawdopodobnie z powodu nieformalnej zmiany kolejności, która utrudnia modelowi precyzyjne cytowanie z poszczególnych pozycji.

### Wnioski końcowe

Wyniki potwierdzają, że context engineering ma istotny wpływ na jakość systemu RAG i że wybór odpowiedniej strategii zależy od domeny i charakteru zapytań. Dla dokumentów prawnych o silnej strukturze formalnej:

- chunking strukturalny przewyższa semantyczny zarówno pod względem jakości, jak i kosztu,
- query rewriting niekoniecznie pomaga, jeśli pytania są już sformułowane formalnym językiem,
- kompresja kontekstu może drastycznie zmniejszyć koszt tokenowy (do 27 razy w przypadku BriefContext) bez znaczącej utraty jakości,
- BriefContext osiąga najlepszy stosunek jakości do kosztu i jest najbardziej skuteczną z testowanych metod mitygacji lost-in-the-middle,
- system jest odporny na halucynacje niezależnie od konfiguracji, prompt systemowy wymuszający odmowę odpowiedzi przy braku informacji w kontekście działa skutecznie,
- prosta konfiguracja baseline (chunking strukturalny, brak rerankingu, brak kompresji) jest zaskakująco mocna i powinna być zawsze rozważana jako punkt odniesienia przy projektowaniu systemów RAG.

Najważniejszym wnioskiem metodologicznym jest konieczność stosowania metryki completeness obok faithfulness i answer relevance. Bez completeness wszystkie konfiguracje uzyskują podobne maksymalne oceny, a istotne różnice w jakości retrievala pozostają niewidoczne.

## Wykorzystane technologie

- **Python 3.12**
- **LangChain**, orkiestracja pipeline'u RAG.
- **Hugging Face Embeddings** (`paraphrase-multilingual-mpnet-base-v2`), reprezentacja semantyczna dokumentów i zapytań.
- **Qdrant**, baza wektorowa i wyszukiwanie semantyczne.
- **Modele LLM** (Groq Llama 3.3 70B, Google Gemini 2.5 Flash/Flash-Lite), modele generatywne odpytywane za pośrednictwem kluczy API.
- **Ragas**, ocena jakości retrievala (context precision, context relevance).
- **NumPy**, obliczenia podobieństwa wektorowego (semantic cache).
- **Streamlit**, aplikacja w formie czatu z możliwością wyboru metod.
- **PyMuPDF / PyPDFLoader**, parsowanie dokumentów PDF.


## Bibliografia

1. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. [LINK](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
2. Reuter et al. (2025). Towards Reliable Retrieval in RAG Systems for Large Legal Datasets. [LINK](https://aclanthology.org/2025.nllp-1.3.pdf)
3. Ma, X. et al. (2023). Query Rewriting for Retrieval-Augmented Large Language Models. [LINK](https://arxiv.org/pdf/2305.14283)
4. Wu, Z. et al. (2022). CONQRR: Conversational Query Rewriting for Retrieval with Reinforcement Learning. [LINK](https://aclanthology.org/2022.emnlp-main.679.pdf)
5. Guo, S. et al. (2025). Enhancing RAG Efficiency with Adaptive Context Compression. [LINK](https://arxiv.org/pdf/2507.22931)
6. Cheng, X. et al. (2024). xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token. [LINK](https://arxiv.org/pdf/2405.13792)
7. Ou, L., & Lapata, M. (2025). Context-Aware Hierarchical Merging for Long Document Summarization. [LINK](https://arxiv.org/pdf/2502.00977)
8. Jin, C. et al. (2024). RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation. [LINK](https://arxiv.org/pdf/2404.12457)
9. Liu, N. F. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. [LINK](https://arxiv.org/pdf/2307.03172)
10. Zhang, G. et al. (2025). A MapReduce Approach to Effectively Utilize Long Context Information in Retrieval Augmented Language Models. [LINK](https://arxiv.org/pdf/2412.15271)
