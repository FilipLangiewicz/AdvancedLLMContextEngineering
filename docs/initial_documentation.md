# NLP - dokumentacja wstępna projektu
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

Kluczowym wyzwaniem współczesnych systemów RAG jest efektywne zarządzanie długim kontekstem - modele językowe dysponują skończonym oknem kontekstowym i nie są w stanie przetworzyć obszernych dokumentów w całości. Projekt skupia się na domenie prawnej, gdzie akty, kodeksy i regulaminy liczą dziesiątki lub setki stron, a precyzja i kompletność odpowiedzi ma szczególne znaczenie. W tym kontekście identyfikujemy następujące problemy:

- **Zarządzanie długim kontekstem:** dokumenty prawne przekraczające 10 000 tokenów wymagają podziału na fragmenty. Standardowy podział stałej długości nie uwzględnia struktury prawnej dokumentu, co prowadzi do rozbicia logicznie powiązanych przepisów i utraty kontekstu semantycznego.
- **Rozumienie zapytań:** pytania użytkowników są często sformułowane w języku potocznym, niedostosowanym do struktury bazy wiedzy, co bez mechanizmu ich przekształcania obniża trafność wyszukiwania.
- **Porównanie strategii zarządzania kontekstem:** brakuje empirycznych podstaw do oceny, która strategia przetwarzania kontekstu - kompresja, hierarchiczne podsumowywanie czy inne - daje najlepsze wyniki w domenie prawnej.
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

Chao Jin i in. (2024) proponują RAGCache – mechanizm cache’owania fragmentów wiedzy i stanów modelu w architekturze hierarchicznej. Praca pokazuje, że takie podejście znacząco redukuje czas do wygenerowania pierwszego tokenu (TTFT) i zwiększa przepustowość [8].

### Problem „lost-in-the-middle”

Nelson F. Liu i in. (2023) wykazują, że modele językowe mają trudności z wykorzystaniem informacji znajdujących się w środkowej części długiego kontekstu [9]. Gongbo Zhang i in. (2025) proponują BriefContext – podejście dzielące kontekst na części i agregujące wyniki (map-reduce), co pozwala lepiej wykorzystać rozproszone informacje i poprawia jakość odpowiedzi [10].


## Opis rozwiązania
Projekt obejmuje modułowy system RAG dla dziedziny prawa energetycznego. Celem jest porównanie różnych technik zaawansowanego zarządzania kontekstem (context engineering) w pracy z obszernymi dokumentami o formalnym charakterze. Projekt został podzielony na pięć sekcji przedstawionych poniżej.

### 1. Dane
Dokumenty PDF wykorzystane do realizacji projektu zostały umieszczone w katalogu `data/pdf` (`document1.pdf`, `document2.pdf`, `document3.pdf`). Zawierają one akty i regulacje prawne dotyczące sektora energetycznego, a w szczególności:

- **Dokument 1:** Ustawa z dnia 20 maja 2016 r. o efektywności energetycznej (33 strony),
- **Dokument 2:** Rozporządzenie Ministra Klimatu z dnia 7 kwietnia 2020 r. w sprawie szczegółowych zasad kształtowania i kalkulacji taryf oraz rozliczeń z tytułu zaopatrzenia w ciepło (23 strony),
- **Dokument 3:** Ustawa z dnia 10 kwietnia 1997 r., Prawo energetyczne (428 stron). 

### 2. Przetwarzanie dokumentów i baza wektorowa
W projekcie zostanie przygotowany pipeline przetwarzania dokumentów obejmujący etapy od wczytania plików PDF, przez czyszczenie treści, po podział tekstu i zapis reprezentacji semantycznych w bazie wektorowej. Porównane zostaną dwa podejścia do chunkingu, aby porównać strategię podziału opartego na strukturze dokumentu ze strategią bazującą na podobieństwie semantycznym treści. Taka organizacja danych ma umożliwić późniejszą analizę wpływu sposobu segmentacji dokumentów na jakość odpowiedzi systemu. Całość zostanie zaprojektowana modułowo, aby możliwe było rozwijanie lub wymiana poszczególnych etapów bez przebudowy całego procesu.

### 3. Pipeline RAG
Pipeline RAG dotyczy etapów prowadzących od analizy zapytania (promptu) do wygenerowania odpowiedzi na podstawie kontekstu z bazy wektorowej. Główne etapy przetwarzania obejmują:
- query understanding - doprecyzowanie treści zapytania, zostanie zaimplementowana strategia rewriting, która przeredaguje zapytanie użytkownika na bardziej formalną wersję odpowiadającą dokumentom prawnym,
- retrieval - pobranie trafnych fragmentów kontekstu z bazy wektorowej za pomocą wyszukiwania semantycznego,
- reranking - uporządkowanie wyników wyszukiwania, dostępne będą dwie opcje: jedna zachowująca domyślną kolejność fragmentów i druga korzystająca z metody "U-shaped", która stanowi jednocześnie metodę mitygacji zjawiska "lost-in-the-middle",
- context compression - skrócenie kontekstu do kluczowych treści zgodnie z wybranymi metodami. Zostanie zaimplementowana metoda filtru ekstrakcyjnego, który usuwa nieistotne fragmenty tekstu oraz metoda hierarchiczna, która podsumowuje każdy chunk, a następnie agreguje do jednego streszczenia. Opcjonalnie zostanie zaimplementowane również podejście map-reduce inspirowane BriefContext, które dzieli pobrane fragmenty na partycje, generuje odpowiedź dla każdej z nich osobno, a następnie agreguje wyniki - stanowi to dodatkową metodę mitygacji zjawiska "lost-in-the-middle", 
- generation - moduł, który buduje finalny kontekst oraz odpytuje model LLM na podstawie zdefiniowanego promptu systemowego, który wymusza odpowiedź ściśle bazującą na przesłanym kontekście.

Analogiczne zdefiniowanie kolejnych kroków i metod prowadzących do wygenerowania najlepszej odpowiedzi zostało przedstawione w schemacie "Rewrite-Retrieve-Read" [3].

Dodatkowo zostanie zastosowana metoda semantic cache, która będzie optymalizowała obsługę powtarzalnych zapytań, które są zbliżone semantycznie. Dla nowego zapytania zostanie wyliczony embedding i będzie on porównany z embeddingami wcześniejszych zapytań za pomocą metryki cosine similarity. Jeśli inne zapytanie jest wystarczająco podobne, na podstawie ustalonego progu, to zostanie zwrócona wygenerowana wcześniej odpowiedź.

### 4. Aplikacja Streamlit
Zostanie przygotowana aplikacja w formie czatu z wykorzystaniem pakietu Streamlit. Interfejs umożliwi zmianę najważniejszych ustawień, aby móc porównać różne metody i strategie. Po przesłaniu zapytania, aplikacja będzie zwracała odpowiedź wraz z dokładnym odwołaniem do źródła.

### 5. Plan eksperymentów i ewaluacji
Istotnym elementem projektu będzie porównanie zaimplementowanych metod w celu zbadania ich wpływu na jakość generowanych odpowiedzi. Zestaw testowy zostanie przygotowany ręcznie na podstawie wiedzy eksperckiej z dziedziny prawa energetycznego, bez zdefiniowanych poprawnych odpowiedzi - ocena jakości będzie opierać się na metodzie "LLM-as-a-Judge", gdzie model językowy ocenia trafność, kompletność i zgodność odpowiedzi z dostarczonymi fragmentami kontekstu. Ewaluacja będzie skupiona na jakości odpowiedzi mierzonej metrykami takimi jak faithfulness i answer relevance, oraz aspektach praktycznych, takich jak stabilność działania na dłuższym kontekście, czas odpowiedzi i koszt przetwarzania. Dodatkowo zostanie przeprowadzony kontrolowany eksperyment badający zjawisko "lost-in-the-middle", polegający na umieszczaniu kluczowego fragmentu na różnych pozycjach w kontekście i mierzeniu wpływu tej pozycji na jakość odpowiedzi.

## Wykorzystane technologie

- **Python 3.12**
- **LangChain** - orkiestracja pipeline'u RAG.
- **Hugging Face Embeddings** - reprezentacja semantyczna dokumentów i zapytań.
- **Qdrant** - baza wektorowa i wyszukiwanie semantyczne.
- **modele LLM** (Google, Groq) - modele generatywne odpytywane za pośrednictwem kluczy API.
- **NumPy** - obliczenia podobieństwa wektorowego (semantic cache).
- **Streamlit** - aplikacja w formie czatu z możliwością wyboru metod.


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