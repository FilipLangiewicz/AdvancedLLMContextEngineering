# Temat projektu: Context Engineering - zaawansowane zarządzanie kontekstem

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

Kluczowym wyzwaniem współczesnych systemów RAG jest efektywne zarządzanie długim kontekstem — modele językowe dysponują skończonym oknem kontekstowym i nie są w stanie przetworzyć obszernych dokumentów w całości. Projekt skupia się na domenie prawnej, gdzie akty, kodeksy i regulaminy liczą dziesiątki lub setki stron, a precyzja i kompletność odpowiedzi ma szczególne znaczenie. W tym kontekście identyfikujemy następujące problemy szczegółowe:

- **Zarządzanie długim kontekstem:** dokumenty prawne przekraczające 10 000 tokenów wymagają podziału na fragmenty. Standardowy podział stałej długości nie uwzględnia struktury prawnej dokumentu, co prowadzi do rozbicia logicznie powiązanych przepisów i utraty kontekstu semantycznego.
- **Rozumienie zapytań:** pytania użytkowników są często sformułowane w języku potocznym, niedostosowanym do struktury bazy wiedzy, co bez mechanizmu ich przekształcania obniża trafność wyszukiwania.
- **Porównanie strategii zarządzania kontekstem:** brakuje empirycznych podstaw do oceny, która strategia przetwarzania kontekstu — kompresja, hierarchiczne podsumowywanie czy inne — daje najlepsze wyniki w domenie prawnej.
- **Efektywność przetwarzania:** brak mechanizmu cache'owania powoduje, że każde zapytanie dotyczące tych samych dokumentów generuje zbędne obciążenie systemu.
- **Zjawisko „lost-in-the-middle":** modele językowe mają tendencję do pomijania informacji z środkowych części kontekstu, co w dokumentach prawnych może skutkować pominięciem kluczowych przepisów lub odesłań.

## Przegląd literatury

### RAG i zarządzanie kontekstem

Retrieval-Augmented Generation zostało wprowadzone przez Patrick Lewis i in. (2020) jako podejście łączące modele językowe z zewnętrznym wyszukiwaniem dokumentów. Autorzy pokazują, że integracja dokumentów źródłowych w procesie generacji poprawia trafność odpowiedzi i ogranicza halucynacje, szczególnie w zadaniach wymagających wiedzy opartej na danych [1]. W kontekście prawa RAG jest szczególnie istotny ze względu na dużą objętość i zmienność aktów prawnych.

### Chunking i reprezentacja dokumentów

Literatura wskazuje, że proste dzielenie tekstu na fragmenty o stałej długości jest niewystarczające dla złożonych dokumentów. Reuter et al. (2025) proponują Summary-Augmented Chunking, w którym każdy fragment jest wzbogacony o globalne streszczenie dokumentu, co poprawia trafność wyszukiwania [2]. Równolegle rozwijane są podejścia semantyczne (dzielenie według treści) i strukturalne (np. według artykułów i paragrafów), które lepiej zachowują spójność logiczną.

### Query rewriting

Xibei Ma i in. (2023) proponują schemat Rewrite-Retrieve-Read, w którym zapytanie użytkownika jest najpierw reformułowane do postaci bardziej zgodnej z językiem dokumentów i wymaganiami wyszukiwarki [3]. Zeqiu Wu i in. (2022) w pracy nad CONQRR pokazują, że przepisywanie zapytań konwersacyjnych do formy niezależnej (stand-alone) poprawia skuteczność wyszukiwania, zwłaszcza w systemach typu chat [4].

### Kompresja kontekstu i podsumowywanie

Shuyu Guo i in. (2025) wprowadzają ACC-RAG, który adaptacyjnie kompresuje kontekst poprzez selekcję najważniejszych tokenów, co pozwala zmniejszyć koszty obliczeniowe bez istotnej utraty jakości [5]. Xin Cheng i in. (2024) proponują xRAG, gdzie dokument reprezentowany jest jako pojedynczy embedding-token, co znacząco redukuje długość wejścia [6]. Z kolei Litu Ou i Mirella Lapata (2025) analizują hierarchiczne podsumowywanie, wskazując, że brak dostępu do oryginalnego kontekstu podczas agregacji zwiększa ryzyko generowania nieścisłości [7].

### Cache w systemach RAG

Chao Jin i in. (2024) proponują RAGCache – mechanizm cache’owania fragmentów wiedzy i stanów modelu w architekturze hierarchicznej. Praca pokazuje, że takie podejście znacząco redukuje czas generacji i zwiększa przepustowość, szczególnie w scenariuszach z powtarzalnymi zapytaniami do tych samych dokumentów [8].

### Problem „lost-in-the-middle”

Nelson F. Liu i in. (2023) wykazują, że modele językowe mają trudności z wykorzystaniem informacji znajdujących się w środkowej części długiego kontekstu [9]. Gongbo Zhang i in. (2025) proponują BriefContext – podejście dzielące kontekst na części i agregujące wyniki (map-reduce), co pozwala lepiej wykorzystać rozproszone informacje i poprawia jakość odpowiedzi [10].


## Opis rozwiązania
Modułowy system RAG dla domeny prawa energetycznego. Cel: porównanie technik context engineering w jednolitych warunkach eksperymentalnych.

### 1. Dane i źródła danych
Dokumenty PDF umieszczone lokalnie w katalogu `data` (`document1.pdf`, `document2.pdf`, `document3.pdf`). Zakres: akty i regulacje prawne dotyczące sektora energetycznego.
- **Dokument 1:** Ustawa z dnia 20 maja 2016 r. o efektywności energetycznej (33 strony),
- **Dokument 2:** Rozporządzenie Ministra Klimatu z dnia 7 kwietnia 2020 r. w sprawie szczegółowych zasad kształtowania i kalkulacji taryf oraz rozliczeń z tytułu zaopatrzenia w ciepło (23 strony),
- **Dokument 3:** Ustawa z dnia 10 kwietnia 1997 r., Prawo energetyczne (428 stron). 

### 2. Pipeline przetwarzania dokumentów
- Pipeline przetwarzania: wczytanie PDF, czyszczenie tekstu, chunking, generacja embeddingów, zapis do bazy wektorowej.
- Dwie strategie podziału dokumentów: chunking strukturalny (rozdział/art./§) oraz semantic chunking (podobieństwo semantyczne).
- Dwie równoległe bazy wektorowe jako podstawa porównań wpływu chunkingu na jakość odpowiedzi.

### 3. Pipeline RAG
- Sekwencja przetwarzania zapytania: query understanding -> retrieval -> reranking -> context compression -> generation.
- Moduł query understanding: porównanie wariantu bazowego z query rewriting.
- Moduł rerankingu: wariant bazowy vs. U-shape reorder jako mechanizm mitygacji lost-in-the-middle.
- Moduł kompresji kontekstu: ekstrakcyjne filtrowanie treści oraz hierarchiczne podsumowywanie.
- Semantic cache: obsługa powtarzalnych i semantycznie podobnych zapytań; redukcja opóźnień odpowiedzi i kosztu obliczeń.

### 4. Aplikacja Streamlit
- Aplikacja webowa Streamlit z interfejsem czatu do konwersacji z systemem.
- Konfigurowalność eksperymentu z poziomu UI: wybór modelu LLM, strategii chunkingu, metody query understanding, wariantu rerankingu i kompresji kontekstu.
- Prezentacja odpowiedzi wraz ze źródłami i metadanymi dokumentów.

### 5. Plan eksperymentów i ewaluacji
- Seria eksperymentów porównawczych dla wielu konfiguracji metod context engineering.
- Kryteria oceny: trafność, kompletność, odporność na lost-in-the-middle, czas odpowiedzi, koszt przetwarzania.
- Forma prezentacji wyników: tabele porównawcze, analiza jakościowa, wnioski dotyczące kompromisu jakość-wydajność.

## Wykorzystane technologie

- **Python 3.12** - język implementacji.
- **LangChain** - orkiestracja pipeline'u RAG.
- **Hugging Face Embeddings** (m.in. `paraphrase-multilingual-mpnet-base-v2`) - reprezentacja semantyczna dokumentów i zapytań.
- **Qdrant** - baza wektorowa i wyszukiwanie semantyczne.
- **PyPDF / loader PDF** - ekstrakcja treści dokumentów źródłowych.
- **LLM providers** (OpenAI, Anthropic, Google, Groq) - warstwa modeli generatywnych i eksperymenty między-modelowe.
- **Pydantic Settings + python-dotenv** - konfiguracja środowiska i kluczy API.
- **NumPy** - obliczenia podobieństwa wektorowego (semantic cache).
- **Pytest / skrypty testowe** - testy komponentowe i integracyjne.
- **Streamlit** - aplikacja demonstracyjna z czatem i panelem wyboru metod.


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