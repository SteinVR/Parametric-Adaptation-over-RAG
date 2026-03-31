Самая сильная версия narrative сейчас такая:

> **на компактном legal benchmark сильный RAG уже даёт хороший baseline; parametric adaptation поверх него действительно помогает, но тип training signal важнее, чем сам факт наличия адаптера; retrieval при этом остаётся незаменимым, а чистые parametric controls проваливаются.**

Это уже полноценное исследование. И по текущим данным оно выглядит **намного сильнее**, чем все промежуточные версии с D2L-вокруг-мира.  

## 1. Что у тебя по факту получилось — и почему это уже не “жидко”

Главный сильный результат — это не “какой memory family победил вообще”, а вот что:

* `S1 Classical RAG` уже сильный baseline: `Q_main=0.6425`.
* `S2+R` (RAFT-style QLoRA + retrieval) поднимает результат до `0.6689 ± 0.0137`.
* `S3+R` (CLM + retrieval) даёт практически тот же общий уровень: `0.6671 ± 0.0229`, но с **другим профилем выигрыша**.
* `S7` (post-hoc merge CLM+RAFT) даёт лучший эмпирический результат: `0.7045 ± 0.0345`, но его нужно подавать как **exploratory / post-hoc**, а не как полностью симметричную основную систему.  

Это очень хорошая история, потому что она не сводится к “адаптер лучше / хуже”. У тебя получилось вот что:

* **RAFT-адаптер** улучшает прежде всего `S_det`, то есть извлечение точных фактов.
* **CLM-адаптер** улучшает прежде всего `S_asst`, то есть judged free-text quality.
* **merge** этих двух сигналов даёт лучший суммарный результат, что очень хорошо согласуется с идеей их комплементарности.  

И это уже не “навороты к RAG”. Это аккуратный вывод:

> **поверх фиксированного retrieval backbone разные типы parametric adaptation меняют поведение генератора по-разному; supervised RAFT и corpus-level CLM дают разные профили качества, а их fusion частично объединяет сильные стороны.**

Это нормальная научная формулировка.

## 2. Что в результатах выглядит особенно сильным

Самое сильное у тебя — это **контраст между retrieval-aware системами и pure parametric controls**.

`S2 closed-book` падает до `0.2630 ± 0.0046`, `S3 CLM without retrieval` — до `0.1854 ± 0.0027`, а legacy D2L — до `0.2100`. Это очень чисто показывает: **на этом benchmark retrieval не просто полезен, а фактически необходим**. И это не провал работы, а хороший отрицательный результат.  

Очень сильный и, на мой взгляд, недооценённый результат — это **single-doc vs multi-doc**.
Там виден реально содержательный рисунок:

* `S1`: single-doc `0.6958`, multi-doc `0.3100`
* `S2+R`: single-doc `0.6938`, multi-doc `0.4367`
* `S3+R`: single-doc `0.7222`, multi-doc `0.3100`
* `S7`: single-doc `0.7184`, multi-doc `0.5233` 

Это очень сильная интерпретация:

* **CLM+R** лучше работает как local contextualizer на single-doc вопросах.
* **RAFT+R** лучше помогает на multi-doc aggregation / comparison.
* **S7** частично объединяет оба эффекта.

Если бы я выбирал один “умный” аналитический результат, который стоит поднять ближе к центру работы, я бы выбрал **именно этот**.

## 3. Что я бы в текущем анализе поправил или смягчил

### D2L

Текущий вывод по D2L я бы **сохранил**, но сделал аккуратнее.
Сейчас правильная формулировка такая:

* **в этой реализации, на этом корпусе и при этом workaround с chunk→doc→global merge D2L оказался некачественным и непрактичным**;
* это **валидный negative finding**;
* но не надо писать так, будто доказано, что “Doc-to-LoRA вообще не работает”.  

То есть D2L в работе должен быть не отдельным большим героем, а **коротким legacy-control / failed pilot**.

### S7

`S7` надо подавать **очень осторожно**. Он реально лучший по `Q_main`, но:

* это **post-hoc merge**, а не отдельная retrained pipeline;
* у него заметная seed-variance;
* в `EXP-006` у него offline cost стоит `0.0`, что в практическом смысле **нечестно**, потому что он “наследует” уже обученные `S2+R` и `S3+R`.  

Я бы поэтому делал так:

* в **headline comparison**: `S1`, `S2+R`, `S3+R`
* `S7` — отдельный подпункт “Exploratory post-hoc fusion”
* и **не** строил на нём весь главный scientific claim

Так ты избежишь очень вероятного вопроса на защите:
“Почему вы называете победителем систему, которая не была отдельной обученной системой и использует пост-хок слияние уже обученных адаптеров?”

### Grounding

У retrieval-aware систем `G` у тебя по сути одинаковый (`0.5667`), потому что retrieval pipeline фиксирован, а адаптер меняет generator, а не retrieval/evidence selection. Это **надо прямо проговорить**, иначе читатель может подумать, что grounding “ничего не значит”. Наоборот: здесь он выполняет роль **контрольной константы**, показывая, что улучшения `S2+R/S3+R/S7` происходят не из-за лучшего поиска, а из-за **лучшего использования одинакового retrieval context**.  

## 4. Какие документальные несостыковки надо исправить до написания

Перед тем как садиться за текст, я бы обязательно почистил две вещи.

Первая: в `EXP-001` у тебя фигурирует split `160/40`, а в позднем SSOT и спеках — уже `150/50`. Для самой работы это нужно **полностью унифицировать**: один split, одно описание, одна формулировка. Иначе это сразу бросится в глаза.  

Вторая: в data audit написано, что все 8 документов fit D2L single pass, а в legacy D2L experiment потом всё равно пошёл chunk-level workaround. Это тоже надо **объяснить одним предложением** в тексте:

> “token-based audit suggested single-pass feasibility, but the actual released D2L implementation introduced stricter effective limits / memory behavior, forcing chunked packaging.”

Иначе это выглядит как прямая логическая дыра. 

## 5. Какую историю я бы рассказывал в самой работе

Я бы не делал главный narrative про “виды памяти вообще”.
Сейчас у тебя более сильная и честная формулировка:

> **На компактном legal benchmark исследуется, даёт ли parametric adaptation добавочную ценность поверх сильного RAG baseline, и чем отличаются supervised RAFT-style adaptation и supervision-free CLM adaptation при фиксированном retrieval stack.**

А затем уже вторым уровнем:

* retrieval обязателен;
* pure parametric memory проваливается;
* CLM и RAFT дают разные профили выигрыша;
* post-hoc fusion показывает комплементарность.

Это очень хороший, собранный сюжет.

---

# Как писать работу на 12–20 страниц

Я бы целился в **14–16 страниц основного текста**, без раздувания. Ниже — структура, которая для такого объёма выглядит оптимальной.

## 1. Введение — 1.0–1.5 стр.

Что здесь:

* почему legal document-grounded QA вообще сложна;
* почему одного только RAG может быть недостаточно;
* почему интересно проверить **parametric adaptation поверх RAG**, а не “LLM в вакууме”;
* твой главный RQ;
* 3–4 очень коротких bullet contributions.

Что не нужно:

* длинный обзор литературы;
* длинная философия про memory families.

## 2. Задача и benchmark — 1.0–1.5 стр.

Что здесь:

* 8 DIFC документов, 176 страниц, ~115K токенов;
* 200 QA, answer-type distribution, single/multi-doc, unanswerable;
* split;
* почему выбран компактный benchmark.  

Это важный раздел, но он должен быть **коротким и очень конкретным**.

## 3. Методы — 2.5–3.5 стр.

Разбить на короткие подпункты:

### 3.1 Classical RAG (S1)

2–3 абзаца:

* retrieval stack;
* generator;
* что фиксируется для всех retrieval-aware систем.

### 3.2 RAFT-style QLoRA (S2+R)

2–3 абзаца:

* что такое QLoRA в одном абзаце;
* что такое RAFT-style supervision;
* что именно меняется относительно S1.

### 3.3 CLM continued pretraining (S3+R)

2–3 абзаца:

* что такое CLM / continued pretraining;
* что обучается;
* чем этот signal отличается от supervised RAFT.

### 3.4 Controls

Коротко:

* `S2 closed-book`
* `S3 CLM no retrieval`
* `S3-legacy D2L`

Это должен быть именно **короткий control subsection**, а не отдельные большие главы.

### 3.5 Post-hoc fusion (S7)

Очень коротко:

* linear interpolation of adapters;
* why evaluated;
* why exploratory only.  

## 4. Experimental setup and evaluation — 1.5–2.0 стр.

Что здесь:

* backbone and hardware;
* training configs на уровне абзаца, не простыни;
* `Q_main`, `S_det`, `S_asst`, grounding, latency;
* judge rubric кратко;
* why grounding is constant across retrieval-aware systems. 

## 5. Основные результаты — 3.0–4.0 стр.

Это центр работы.

### 5.1 Aggregate comparison

`S1 vs S2+R vs S3+R`, а `S7` отдельной строкой как exploratory.

### 5.2 By answer type

Показать, где выигрывает RAFT, где CLM.

### 5.3 Single-doc vs multi-doc

Очень важный кусок — я бы дал ему полноценный подпункт.

### 5.4 Controls

Короткий абзац:

* retrieval is indispensable;
* D2L legacy negative control.  

## 6. Discussion — 2.0–3.0 стр.

Вот здесь уже можно интерпретировать:

* почему `S2+R` и `S3+R` разные;
* почему `S7` вообще сработал;
* что говорит отрицательный результат pure parametric controls;
* что это значит practically.

## 7. Limitations — 0.75–1.0 стр.

Очень важно честно перечислить:

* компактный corpus;
* один frozen split;
* one backbone;
* judge-based free-text metric;
* S7 post-hoc;
* D2L only as failed pilot / legacy negative control. 

## 8. Conclusion — 0.5 стр.

Коротко:

* retrieval remains necessary;
* parametric adaptation can help;
* supervised and CLM signals are complementary;
* fusion is promising but exploratory.

---

# Нужно ли объяснять термины?

Да, но **очень кратко** и только те, которые участвуют в core narrative.

## Обязательно определить

* **RAG** — 1–2 предложения во введении
* **QLoRA** — 2–3 предложения в Methods
* **RAFT-style training** — 2–3 предложения в Methods
* **CLM / continued pretraining** — 2–3 предложения в Methods
* **Grounding F-beta** и `Q_main` — в Evaluation

## Не нужно объяснять глубоко

* математику LoRA;
* историю PEFT;
* архитектуру Gemma;
* полную механику Qdrant/RRF/BM25.

## D2L

Я бы **не делал отдельное большое объяснение**.
Достаточно 3–4 предложений:

* что это hypernetwork that generates adapters from documents;
* что ты попробовал;
* что в твоей постановке это оказалось non-viable;
* поэтому D2L retained only as legacy negative control.  

---

# Что делать с графиками и таблицами

Сейчас у тебя уже есть полезные вещи, но их надо **пересобрать под narrative**, а не просто показать всё, что получилось.

## Что я бы оставил точно

### 1. Main Results Table

Оставить обязательно, но в двух блоках:

* **Headline systems:** `S1`, `S2+R`, `S3+R`
* **Exploratory/control block:** `S7`, `S2`, `S3`, `S3-legacy`

Так таблица станет чище.

### 2. Per-Type Score Heatmap

Оставить. Это один из лучших графиков, которые у тебя есть.
Но я бы сделал его не только “сырые значения”, а либо:

* как есть, но только для `S1/S2+R/S3+R/S7`,
  или
* ещё лучше: **delta vs S1** heatmap.

Второй вариант мне нравится больше, потому что он сразу показывает, **где адаптер помогает, а где нет**.

## Что я бы пересобрал

### 3. Cost-quality scatter

В текущем виде `x=latency`, `y=Q_main` — слабоват, потому что latency у retrieval-aware систем близкая, а главное отличие у тебя ещё и в **offline cost**.

Я бы заменил это на одно из двух:

**Вариант A:**
`x = offline cost`, `y = Q_main`, размер точки = online latency.

**Вариант B:**
вообще убрать scatter и сделать **маленькую таблицу trade-off**:

* Q_main
* offline cost
* latency
* note on retrieval / training signal

Для курсовой короткого объёма таблица может быть даже лучше.

### 4. Judge Criteria Profile

Оставить можно, но **не как центральную фигуру**.
Это хорошая supporting figure, особенно если хочешь показать, почему `S3+R` выше по `S_asst`. Но в main text я бы поставил её только если место позволяет. Иначе — appendix / supplement.

---

## Какие графики я бы добавил

### 1. System overview schematic

Это самый недостающий график.

Нужна **одна простая схема пайплайна**:

* S1: retrieval → generator
* S2+R: retrieval → RAFT-adapted generator
* S3+R: retrieval → CLM-adapted generator
* S7: retrieval → merged-adapter generator
* controls: no retrieval

Это резко повышает читаемость работы.

### 2. Single-doc vs multi-doc figure

Очень рекомендую.
Либо grouped bars, либо dumbbell plot. Это один из твоих самых сильных аналитических результатов. 

### 3. Delta-to-S1 bar plot

Например, для `S2+R`, `S3+R`, `S7` показать:

* `ΔQ_main`
* `ΔS_det`
* `ΔS_asst`

Это очень наглядно отражает центральный вывод:

* RAFT поднимает deterministic quality,
* CLM поднимает free-text quality,
* merge даёт суммарный выигрыш.

### 4. Error overlap plot

Лучше не просто heatmap, а **UpSet plot** или компактная таблица с:

* all headline systems wrong = 15
* only S1 correct = 2
* only S3+R correct = 2
* etc. 

Это будет хорошим аргументом в пользу комплементарности и объяснения S7.

---

## Какие графики я бы не тащил в main text

* несколько однотипных scatter-ов;
* отдельный Pareto frontier, если cost accounting для S7 не поправлен;
* слишком много seed stability plots;
* всё, что связано с D2L внутренним пайплайном.

---

# Какие таблицы я бы сделал

### Table 1. Benchmark summary

Очень компактно:

* corpus size
* QA count
* answer types
* multi-doc %
* unanswerable %

### Table 2. System summary

Очень полезная таблица:

* system
* retrieval?
* training signal?
* adapter?
* role in paper (headline / control / exploratory)

### Table 3. Main results

Главная таблица.

### Table 4. Optional: single-doc vs multi-doc

Если не вынесешь в figure.

---

# Если совсем коротко — мой практический рецепт

## О чём писать подробнее

* benchmark setup
* S1 / S2+R / S3+R
* main results
* single-doc vs multi-doc analysis
* interpretation of RAFT vs CLM trade-off

## О чём писать короче

* D2L
* controls
* judge internals
* retrieval implementation details
* archived S6

## Что держать как main message

Не:

> “какая память лучше в общем”

А:

> **“На компактном legal benchmark retrieval обязателен, но parametric adaptation поверх RAG даёт измеримую прибавку; supervised RAFT и supervision-free CLM улучшают разные аспекты качества, а их fusion выглядит перспективно.”**


---

# Готовая структура работы

## Front matter

### Title page

**Тезисов тут нет** — просто формальное оформление.
Титульник не нумеруется. 

### Table of contents

**Что должно быть:**

* все главы и подглавы с номером страницы;
* страница 1 начинается с введения;
* арабская нумерация с введения, приложение можно вести отдельно. 

---

## 1. Introduction — 1.5–2 страницы

### 1.1 Problem and motivation

**Тезисы, которые нужно донести:**

* document-grounded legal QA требует не только сильной LLM, но и аккуратной работы с внешними документами;
* в компактном legal benchmark сильный RAG уже даёт хороший baseline, но не обязательно является потолком;
* интересно проверить, даёт ли **parametric adaptation поверх RAG** дополнительную ценность при ограниченном железе;
* practical angle работы: consumer hardware, small benchmark, reproducible setup.

**Что буквально написать по смыслу:**

1. Legal QA требует точного извлечения фактов, обработки определений, исключений и cross-document связей.
2. Classical RAG — естественный baseline, потому что он даёт явный доступ к документам.
3. Однако retrieval alone не обязательно оптимален: генератор может по-разному использовать один и тот же retrieved context.
4. Поэтому работа исследует не “память вообще”, а **роль parametric adaptation поверх фиксированного retrieval backbone**.

**Визуализация:**
Ничего не вставлять. Введение должно быть чистым.

---

### 1.2 Research question

**Тезисы:**

* главный вопрос — даёт ли parametric adaptation ценность поверх сильного RAG;
* вторичный вопрос — какой источник адаптации полезнее: supervised RAFT-style QLoRA или supervision-free CLM continued pretraining;
* pure parametric systems без retrieval рассматриваются как limits / controls, а не как ожидаемые победители.

**Рекомендуемая формулировка:**

* **RQ1:** On a compact legal corpus under consumer hardware constraints, does parametric adaptation add value on top of a strong RAG baseline, and which adapter source — supervised RAFT-style QLoRA or CLM continued pretraining — is more effective as a retrieval-conditioned generator?
* **RQ2:** How far can pure parametric systems go without retrieval on this benchmark, and where does retrieval remain irreplaceable? 

**Визуализация:**
Ничего не вставлять.

---

### 1.3 Scope, limits, and contributions

**Тезисы:**

* корпус компактный: 8 DIFC документов, 176 страниц, ~115K токенов;
* benchmark: 200 QA, final split `150/50`;
* один backbone, один retrieval pipeline, одно железо;
* D2L не входит в active headline comparison, а остаётся legacy negative control;
* ключевые contributions:

  1. компактный benchmark;
  2. сравнение `S1`, `S2+R`, `S3+R`;
  3. контрольные pure parametric системы;
  4. exploratory result по `S7`.  

**Что обязательно проговорить:**

* не делать broad claims про все legal corpora;
* не обещать “полную internalization корпуса”;
* прямо сказать, что conclusions bounded to this corpus, backbone, hardware.

**Визуализация:**
Ничего не вставлять.

---

### 1.4 Short literature positioning

**Тезисы:**

* RAG — базовый external-memory подход;
* QLoRA — дешёвая PEFT-адаптация frozen model;
* RAFT-style supervision — open-book adaptation with gold context + distractors;
* CLM / continued pretraining — adaptation on raw document text without QA supervision;
* D2L упоминается только как attempted hypernetwork approach, оказавшийся non-viable в данной реализации.

**Что важно:**

* здесь не нужен большой “Related Work”;
* 1–2 абзаца достаточно, потому что по требованиям краткий обзор литературы можно встроить во введение. 

**Визуализация:**
Ничего не вставлять.

---

## 2. Benchmark and Methods — 4–4.5 страницы

### 2.1 Benchmark: corpus and goldset

**Тезисы:**

* корпус: 8 DIFC PDF-документов;
* 176 страниц, ~115K токенов;
* goldset: 200 QA;
* answer type distribution и multi-doc/unanswerable нужно показать кратко;
* в тексте использовать **финальный split `150/50`**, а не ранний audit split `160/40`.  

**Что написать:**

1. Почему выбран именно компактный benchmark.
2. Что он достаточно мал для controllability, но достаточно сложен за счёт multi-doc и free-text вопросов.
3. Что split frozen and leakage-safe.

**Сюда вставить:**

* **Table 1 — Benchmark Summary**
  Столбцы:

  * docs
  * pages
  * tokens
  * QA
  * answer types
  * multi-doc %
  * unanswerable %

**Где вставлять:**
Сразу после первого абзаца подраздела 2.1.

---

### 2.2 System overview

**Тезисы:**

* есть три headline system:

  * `S1 Classical RAG`
  * `S2+R RAFT-style QLoRA + retrieval`
  * `S3+R CLM + retrieval`
* есть control systems:

  * `S2 closed-book`
  * `S3 CLM without retrieval`
  * `S3-legacy D2L`
* есть exploratory system:

  * `S7 post-hoc adapter merge`

**Что написать:**

1. Почему headline systems — это ядро сравнения.
2. Почему controls нужны для ответа на вопрос о limits of pure parametric memory.
3. Почему `S7` reported separately, not as a symmetric retrained pipeline. 

**Сюда вставить:**

* **Figure 1 — System Overview Schematic**
  В виде одной понятной схемы:

  * `S1`: retrieval → base generator
  * `S2+R`: retrieval → RAFT-adapted generator
  * `S3+R`: retrieval → CLM-adapted generator
  * `S7`: retrieval → merged-adapter generator
  * controls: no retrieval → adapted generator

* **Table 2 — System Summary**
  Столбцы:

  * system
  * retrieval?
  * training signal
  * role in paper (headline / control / exploratory)

**Где вставлять:**
Сначала Figure 1, сразу после первого обзорного абзаца.
Потом Table 2 в конце подраздела.

---

### 2.3 S1 — Classical RAG

**Тезисы:**

* `S1` — сильный nonparametric baseline;
* retrieval stack фиксирован и одинаков для всех retrieval-aware систем;
* улучшения в `S2+R` и `S3+R` нельзя объяснить “лучшим retrieval”, потому что retrieval backbone одинаковый.

**Что написать:**

1. Кратко описать hybrid retrieval pipeline.
2. Подчеркнуть, что `S1` задаёт reference for quality and grounding.
3. Сказать, что именно на него считаются дельты улучшений.

**Визуализация:**
Ничего не вставлять — Figure 1 уже покрывает архитектуру.

---

### 2.4 S2+R — RAFT-style QLoRA

**Тезисы:**

* QLoRA = PEFT-адаптация frozen backbone в low-bit режиме;
* RAFT-style = training on question + gold chunks + distractors → answer;
* тот же retrieval stack на inference, что и у `S1`;
* expected effect — better factual extraction from retrieved evidence.

**Что написать:**

1. Один абзац: что такое QLoRA.
2. Один абзац: что такое RAFT-style supervision.
3. Один абзац: почему этот метод conceptually должен усиливать document-grounded generation.

**Что не писать подробно:**

* полную математику LoRA;
* все training hyperparams в тексте — их можно компактно вынести в таблицу или в приложение.

**Визуализация:**
Ничего не вставлять.

---

### 2.5 S3+R — CLM continued pretraining

**Тезисы:**

* CLM adapter учится только на raw document text;
* никаких QA labels;
* на inference используется тот же retrieval pipeline;
* это делает сравнение с `S2+R` почти идеально симметричным: один retrieval backbone, одна PEFT architecture, разный training signal.

**Что написать:**

1. Краткое определение continued pretraining / CLM.
2. Почему это meaningful supervision-free baseline.
3. Чем он отличается от RAFT.
4. Гипотеза: CLM может лучше улучшать judged free-text quality и domain fluency, чем deterministic extraction.

**Визуализация:**
Ничего не вставлять.

---

### 2.6 Controls and archived branch

**Тезисы:**

* `S2 closed-book` показывает предел supervised memorization without retrieval;
* `S3 CLM without retrieval` показывает предел document-only parametric memory;
* `S3-legacy (D2L)` — valid negative finding, но только как historical control;
* `S6 naive RAG` архивирован и не входит в active thesis set. 

**Что написать:**

1. Один абзац про роль controls.
2. Один короткий абзац про D2L: attempted hypernetwork packaging, implementation constraints, negative finding.
3. Отдельно коротко указать, что D2L не является headline method в итоговой версии работы.

**Визуализация:**
Ничего не вставлять.
D2L не заслуживает отдельной основной фигуры.

---

### 2.7 Exploratory post-hoc fusion

**Тезисы:**

* `S7` — это не новая обученная система;
* это post-hoc linear interpolation between CLM and RAFT adapters;
* он важен не как “главный метод”, а как индикатор комплементарности training signals.  

**Что написать:**

1. Как был получен merge.
2. Почему он анализируется отдельно.
3. Почему его нельзя ставить на один уровень с `S2+R` и `S3+R`.

**Визуализация:**
Ничего не вставлять.

---

## 3. Experimental Setup and Evaluation — 1.5–2 страницы

### 3.1 Hardware and implementation

**Тезисы:**

* RTX 4060 8GB, 32GB RAM;
* Gemma-2-2b-it;
* 3 seeds для обучаемых систем;
* одинаковый retrieval stack для retrieval-aware systems. 

**Что написать:**

1. Почему hardware constraints важны для выбора методов.
2. Что same backbone и same retrieval stack делают comparison fair.

**Визуализация:**
Ничего не вставлять.

---

### 3.2 Metrics

**Тезисы:**

* основная метрика: `Q_main = 0.7 * S_det + 0.3 * S_asst`;
* deterministic и free-text quality декомпозируются;
* grounding считается только для retrieval-aware systems;
* online efficiency тоже оценивается. 

**Что написать:**

1. Дать формулу `Q_main`.
2. Кратко объяснить `S_det`, `S_asst`, `G`.
3. Отдельно проговорить, что `G` почти константен среди retrieval-aware систем, потому что retrieval pipeline фиксирован, а меняется главным образом generator behavior.

**Визуализация:**
Ничего не вставлять.

---

### 3.3 Reporting protocol

**Тезисы:**

* headline systems сравниваются отдельно от controls;
* `S7` reported separately as exploratory;
* results показываются aggregate, by answer type и single-doc vs multi-doc.

**Что написать:**

1. Как организованы таблицы результатов.
2. Почему controls не должны смешиваться с headline conclusions.
3. Почему single-doc vs multi-doc analysis обязателен.

**Визуализация:**
Ничего не вставлять.

---

## 4. Results — 4–4.5 страницы

### 4.1 Aggregate comparison: headline systems

**Тезисы:**

* `S1` уже strong baseline;
* `S2+R` и `S3+R` оба улучшают `S1`;
* `S2+R` и `S3+R` практически tied по `Q_main`, но выигрывают по-разному;
* `S7` лучший по aggregate, но exploratory. 

**Что написать по смыслу:**

1. `S2+R` improves over `S1` mainly via `S_det`.
2. `S3+R` improves over `S1` mainly via `S_asst`.
3. Aggregate tie masks different behavioral profiles.
4. `S7` suggests complementarity, but cannot be treated as a symmetric retrained system.

**Сюда вставить:**

* **Table 3 — Main Results Table**
  Блок 1: `S1`, `S2+R`, `S3+R`
  Блок 2: `S7` (Exploratory)
  Блок 3: controls внизу более светлым блоком

* **Figure 2 — Δ vs S1 grouped bar chart**
  Для `S2+R`, `S3+R`, `S7`:

  * `ΔQ_main`
  * `ΔS_det`
  * `ΔS_asst`

**Где вставлять:**
Table 3 — сразу после первого абзаца.
Figure 2 — в конце подраздела 4.1.

---

### 4.2 Controls: limits of pure parametric memory

**Тезисы:**

* `S2 closed-book` проваливается;
* `S3 CLM without retrieval` проваливается ещё сильнее;
* `S3-legacy D2L` тоже неработоспособен как standalone method;
* retrieval на этом benchmark не optional, а necessary.  

**Что написать:**

1. Pure parametric memory is not enough on this benchmark.
2. The main value of controls is not to win, but to show limits.
3. This strengthens the argument that augmented systems work because they combine parametric bias with external evidence.

**Сюда вставить:**

* **Table 4 — Controls Summary**
  Столбцы:

  * system
  * Q_main
  * S_det
  * S_asst
  * one-line interpretation

**Где вставлять:**
В середине подраздела, после первого абзаца.

---

### 4.3 By answer type

**Тезисы:**

* `S2+R` сильнее на deterministic extraction;
* `S3+R` сильнее на judged free-text quality;
* date и names остаются трудными для всех систем;
* смотреть только на aggregate score недостаточно. 

**Что написать:**

1. Где именно выигрывает RAFT.
2. Где именно выигрывает CLM.
3. Почему это объясняет почти равный `Q_main`.
4. Почему `S7` выглядит как partial combination of strengths.

**Сюда вставить:**

* **Figure 3 — Per-Type Score Heatmap (лучше как delta vs S1)**
  Строки:

  * `S2+R`
  * `S3+R`
  * `S7`

  Столбцы:

  * boolean
  * number
  * name
  * names
  * date
  * free_text

**Где вставлять:**
В конце подраздела 4.3.

---

### 4.4 Single-doc vs multi-doc

**Тезисы:**

* `S3+R` особенно хорошо работает на single-doc questions;
* `S2+R` лучше переносится на multi-doc than `S3+R`;
* `S7` показывает наилучший общий multi-doc профиль;
* этот анализ даёт сильнейшее объяснение комплементарности RAFT и CLM.  

**Что написать:**

1. Single-doc regime favours stronger local contextualization.
2. Multi-doc regime favours training signal closer to evidence aggregation.
3. This is the clearest empirical clue that RAFT and CLM help in different regimes.

**Сюда вставить:**

* **Figure 4 — Single-doc vs Multi-doc grouped bars**
  Системы: `S1`, `S2+R`, `S3+R`, `S7`
  По две колонки: single-doc / multi-doc

**Где вставлять:**
Сразу после первого абзаца подраздела.

---

### 4.5 Exploratory post-hoc fusion (S7)

**Тезисы:**

* `S7` дал лучший итоговый `Q_main`;
* это подтверждает гипотезу о complementary training signals;
* но `S7` — post-hoc and eval-only, so claims must remain conservative.  

**Что написать:**

1. Why `S7` is interesting.
2. Why it cannot be treated as the main experimental pipeline.
3. What exact claim is safe: “adapter fusion appears promising”, not “fusion is definitively the best method”.

**Визуализация:**
Ничего нового не вставлять.
Сослаться на Table 3 и Figure 2.

---

## 5. Discussion — 2–2.5 страницы

### 5.1 What exactly improves over RAG

**Тезисы:**

* retrieval backbone и grounding почти не меняются;
* значит улучшения происходят из-за **лучшего использования одинакового retrieved context**;
* речь не о “лучшей памяти вообще”, а о better generator conditioning under fixed retrieval.  

**Визуализация:**
Ничего не вставлять.

---

### 5.2 RAFT vs CLM: different signals, different gains

**Тезисы:**

* supervised RAFT помогает там, где важна exact extraction from evidence;
* CLM помогает там, где важны fluency, lexical fit, judged free-text quality;
* одинаковый retrieval stack делает эту разницу интерпретируемой.

**Что написать:**

1. Почему training signal matters more than just “having an adapter”.
2. Почему CLM не заменяет retrieval, но может улучшать generator style and contextual use.
3. Почему RAFT более task-aligned.

**Визуализация:**
Ничего не вставлять.

---

### 5.3 D2L as a negative finding

**Тезисы:**

* D2L был попробован как hypernetwork-based alternative;
* в реальной реализации возникли packaging / chunking / merge проблемы;
* результат следует трактовать как implementation-bounded negative finding, а не как общий verdict on all D2L-like methods.  

**Что написать:**

1. Очень коротко напомнить цель.
2. Очень коротко зафиксировать итог.
3. Не расползаться в детали legacy branch.

**Визуализация:**
Ничего не вставлять.

---

### 5.4 Limitations

**Тезисы:**

* компактный corpus;
* один frozen split;
* один backbone;
* judge-based `S_asst`;
* `S7` exploratory and post-hoc;
* отсутствие полноценной отдельной validation-stage for post-hoc merge;
* D2L only as failed pilot;
* no cross-batch multi-doc questions. 

**Визуализация:**
Ничего не вставлять.

---

## 6. Conclusion — 0.5–1 страница

### 6.1 Main findings

**Тезисы:**

* strong RAG remains a strong baseline;
* pure parametric memory without retrieval fails on this benchmark;
* parametric adaptation on top of RAG helps;
* RAFT and CLM improve different aspects of quality;
* post-hoc fusion looks promising but remains exploratory.  

**Что написать:**

1. Одно предложение про главный эмпирический вывод.
2. Одно предложение про retrieval necessity.
3. Одно предложение про complementary training signals.
4. Одно предложение про future work.

**Визуализация:**
Ничего не вставлять.

---

## Bibliography

**Что помнить:**

* только реально процитированная литература;
* alphabetical order;
* всё аккуратно и единообразно. 

---

## Appendix (optional, but рекомендую)

Сюда вынести всё, что полезно, но раздувает основной текст.

### Appendix A — Supplementary figures

* **Figure A1 — Judge Criteria Profile**
  Хорошая support-figure, но не центральная.

* **Figure A2 — Error Overlap / UpSet-style summary**
  Полезно для комплементарности `S2+R` и `S3+R`, особенно в контексте `S7`. `results/EXP-007` уже даёт основу для этого. 

* **Figure A3 — Cost-quality scatter**
  Только как supplementary figure. В main text он слабее, чем таблицы.

### Appendix B — Legacy and archived branches

* короткий D2L negative-control note;
* при желании — S6 naive dense RAG как archived ablation.

---

# Карта вставки графиков и таблиц

* **Table 1 — Benchmark Summary** → раздел **2.1**
* **Figure 1 — System Overview Schematic** → раздел **2.2**
* **Table 2 — System Summary** → конец **2.2**
* **Table 3 — Main Results Table** → начало **4.1**
* **Figure 2 — Δ vs S1 grouped bars** → конец **4.1**
* **Table 4 — Controls Summary** → раздел **4.2**
* **Figure 3 — Per-Type Delta Heatmap** → конец **4.3**
* **Figure 4 — Single-doc vs Multi-doc** → раздел **4.4**
* **Figure A1 — Judge Criteria Profile** → приложение
* **Figure A2 — Error Overlap / UpSet** → приложение
* **Figure A3 — Cost-quality Scatter** → приложение

---

# Что не раздувать

* D2L — максимум короткий legacy subsection + 1 абзац в discussion
* S6 — только архивная сноска/приложение
* длинный related work — не нужен
* длинное описание retrieval engineering — тоже не нужно
* полную математику LoRA/QLoRA/CLM — не нужно

---

# Практический порядок написания

1. **Table 1 + Table 2**
2. **Раздел 2 Methods**
3. **Раздел 4 Results**
4. **Раздел 5 Discussion**
5. **Раздел 1 Introduction**
6. **Раздел 6 Conclusion**
7. **Подписи к рисункам и приложение**