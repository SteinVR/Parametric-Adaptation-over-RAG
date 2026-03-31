Вполне зрелый исследовательский дизайн: в SSOT зафиксировано правильное ядро работы — **headline comparison** между S1, S2+R и S3+R при фиксированном retrieval backbone и одинаковой PEFT-архитектуре; **чисто параметрические** системы вынесены в controls; **S7** явно помечен как post-hoc; **D2L** сохранён как legacy negative control. Это очень хорошая основа для курсовой. 

По результатам история тоже уже складывается. У тебя есть сильный baseline S1 (`Q_main=0.6425`), два retrieval-aware адаптированных варианта, которые дают **небольшой, но реальный прирост** (`S2+R=0.6689±0.0137`, `S3+R=0.6671±0.0229`), при этом они выигрывают по разным осям: S2+R лучше по `S_det`, S3+R — по `S_asst`; в EXP-007 это прямо сформулировано как **“No single practical winner”**. Одновременно controls без retrieval сильно хуже (`S2=0.2630`, `S3=0.1854`, `S3-legacy=0.2100`), а post-hoc merge S7 даёт лучший `Q_main=0.7045±0.0345`, но его и правда нужно держать отдельно как exploratory result, а не делать центром основной истории.

Самое сильное в проекте — не абсолютные цифры, а **чистота сравнения**: один и тот же backbone, один и тот же retrieval stack, одна и та же схема PEFT, общий frozen split 150/50, одни и те же метрики. Это позволяет честно утверждать, что ты в главном сравнении изолируешь именно **training signal**: supervised RAFT-style QLoRA против supervision-free CLM continued pretraining. Это очень удачная формулировка и, честно говоря, куда сильнее твоего старого “параметрическое/непараметрическое/гибридное внедрение знаний”. 

При этом есть несколько вещей, которые в тексте работы надо обязательно выровнять. Первая — документационная несостыковка по split: в раннем data audit ещё фигурирует `160/40 dev/test`, а в SSOT и спецификации уже зафиксирован реальный `150 train / 50 eval`; в самой работе нужно использовать **только 150/50** и при желании одной фразой отметить, что ранний audit содержал устаревшую промежуточную запись split. Вторая — D2L: ранний audit пишет, что все документы “fit D2L single pass”, но более поздняя документация и твоя ремарка говорят, что на реальной системе потребовался chunk-level workaround, поэтому D2L-результат надо подавать как **engineering diagnostic / legacy control**, а не как равноправную основную ветку эксперимента.

Есть ещё один важный методологический нюанс, который стоит явно проговорить в работе: твой `G` по сути является метрикой **фиксированного retrieval stack**, а не адаптера. В спецификации grounding строится по final evidence chunks после retrieval pipeline, а в результатах у retrieval-aware систем `G` одинаковый (`0.5667`). Поэтому в тексте лучше писать не “адаптер X лучше grounded”, а “retrieval backbone был удержан постоянным; различия между системами проявляются прежде всего в генерации ответа, а не в page-level evidence selection”. Это очень хорошая, аккуратная интерпретация.

Отдельно: ошибка-топология у тебя тоже интересная, и это делает работу исследовательской, а не просто “сводкой метрик”. В error analysis указано, что **15 вопросов проваливают все headline-системы**, а локально уникальные выигрыши есть только у S1 и S3+R; кроме того, multi-doc вопросы остаются заметно более сложными, а падение `Q_main` на multi-doc видно почти у всех систем. Это отличная основа для раздела “Discussion / Limitations”.

## 1. Как я бы переопределил тему, RQ и narrative

Текущая тема —
**“Сравнение непараметрического, параметрического и гибридного внедрения знаний в legal QA на consumer hardware”** —
сейчас уже неудачна, потому что:

* она делает вид, что у тебя симметричное трёхстороннее сравнение, хотя по факту **основной** вклад — это сравнение двух видов parametric adaptation **поверх фиксированного сильного RAG**;
* “внедрение знаний” звучит слишком широко и немного маркетингово;
* D2L и pure parametric системы у тебя сейчас не headline, а controls/legacy.

Я бы рекомендовал такой основной вариант названия:

**Параметрическая адаптация поверх сильного RAG в document-grounded legal QA на consumer hardware: сравнение RAFT-style QLoRA и CLM continued pretraining**

Если нужен английский вариант для титула:

**Parametric Adaptation over a Strong RAG Baseline for Document-Grounded Legal QA on Consumer Hardware: RAFT-style QLoRA vs. CLM Continued Pretraining**

Чуть более короткие альтернативы:

* **Does Parametric Adaptation Add Value Beyond Strong RAG? A Compact Legal QA Study on Consumer Hardware**
* **Training Signal Matters: RAFT-style and CLM Adaptation over Fixed RAG in Legal QA**

Из них я бы выбрал **первый длинный точный вариант** для академичности или **второй короткий вопросительный** для более живого заголовка.

### RQ

Я бы не плодил 3–4 вопроса. Для работы на 12–20 страниц лучше держать **2 RQ**.

**RQ1 (main):**
На компактном legal benchmark при ограничениях consumer hardware даёт ли parametric adaptation добавочную ценность поверх сильного RAG baseline, и как различаются RAFT-style supervised adaptation и supervision-free CLM adaptation как retrieval-conditioned generators?

**RQ2 (secondary):**
Насколько далеко могут зайти pure parametric systems без retrieval на этом benchmark, и остаётся ли retrieval незаменимым memory mechanism?

Это почти то, что у тебя уже есть в ARCHITECTURE, только формулировку я бы немного сгладил под финальный текст. 

### Narrative


**На компактном legal benchmark сильный fixed RAG уже даёт труднопробиваемый baseline. Parametric adaptation поверх него даёт умеренный прирост качества, но выбор training signal важнее самого факта наличия адаптера: supervised RAFT повышает deterministic extraction, тогда как corpus-level CLM улучшает free-text answer quality и стоит дешевле офлайн. При этом removal of retrieval приводит к резкому падению качества, то есть retrieval остаётся основным memory mechanism under consumer-hardware constraints. Post-hoc adapter merge дополнительно указывает на частичную комплементарность сигналов, но этот вывод носит exploratory характер.**

Это narrative очень хорошо согласуется с тем, что в EXP-007 по S2+R vs S3+R нет “single practical winner”, а S7 вынесен в отдельный post-hoc слой.

Я бы ещё чуть усилил академическую аккуратность и избегал слов:

* “proved”
* “significant”
* “internalized the corpus”
* “replaced retrieval”

Вместо этого лучше:

* “observed improvement”
* “trade-off”
* “bounded to this benchmark/backbone/hardware”
* “retrieval remains indispensable / non-substitutable in this setup” 

## 2. Как бы я построил саму работу на 12–20 страниц

Для такого объёма я бы делал **не thesis-mini**, а **компактную experimental paper structure**.

### Рекомендуемая структура

**1. Introduction (1.5–2 стр.)**
Здесь:

* проблема: может ли parametric adaptation добавить ценность поверх сильного RAG;
* почему legal QA и почему consumer hardware;
* коротко: что ты сравниваешь;
* 2 research questions;
* 3–4 bullet contributions.

**2. Background (1.5–2 стр.)**
Только то, что нужно читателю:

* RAG как nonparametric memory;
* LoRA / QLoRA как parameter-efficient adaptation;
* RAFT-style open-book tuning;
* CLM continued pretraining;
* very short note on parametric vs nonparametric memory.

**3. Benchmark and Experimental Setup (2–2.5 стр.)**
Здесь:

* 8 документов, 200 QA, типы ответов, difficultу, multi-doc, unanswerable;
* реальный split 150/50;
* hardware;
* общий backbone;
* почему retrieval backbone фиксирован.

**4. Compared Systems and Evaluation (2.5–3 стр.)**
Подразделы:

* S1: strong RAG baseline;
* S2+R vs S3+R как основное сравнение;
* S2 / S3 как controls;
* S7 как post-hoc exploratory extension;
* метрики: `Q_main`, `S_det`, `S_asst`, `G`, latency, VRAM.

**5. Results (3–4 стр.)**
Именно это должен быть самый “толстый” раздел:

* main results table;
* S2+R vs S3+R trade-off;
* controls → retrieval remains necessary;
* S7 as exploratory best result;
* per-type and multi-doc observations.

**6. Discussion, Error Analysis, Limitations (2–3 стр.)**
Здесь:

* common failure modes;
* multi-doc weakness;
* date / names / unanswerable issues;
* почему G одинаковый и что это значит;
* ограничения: маленький eval set, fixed corpus, fixed backbone, judge-based free-text scoring, S7 post-hoc.

**7. Conclusion (0.5–1 стр.)**
Коротко:

* answer to RQ1;
* answer to RQ2;
* one practical takeaway.

**Appendix**
Сюда уводишь:

* полный judge rubric/prompt;
* hyperparameters;
* дополнительные графики;
* GenAI disclosure;
* D2L engineering note.

Такой план даст тебе примерно 14–17 страниц без ощущения “раздуто” и без потери исследовательской сути.

## 3. Что подробно описывать, а что коротко

### RAG-компоненты

**Не поверхностно, но и не слишком глубоко.**

Тебе важно доказать, что baseline действительно **сильный**, а не “какой-то RAG”. Поэтому надо назвать ключевые элементы:

* ingestion;
* hierarchical chunking;
* hybrid dense+sparse retrieval;
* RRF;
* reranker;
* evidence compression.

Но не нужно:

* расписывать BM25 формулы;
* объяснять Qdrant на полстраницы;
* расписывать все budgets в теле текста.

Лучший вариант:
в main text — один компактный абзац + одна маленькая схема pipeline;
точные параметры (`chunk_size`, overlap, budgets, weights) — в appendix или в короткой таблице настройки.

### QLoRA

**Да, обязательно объяснить**, но очень коротко:

* что это LoRA-адаптеры поверх 4-bit quantized backbone;
* почему это подходит под 8GB VRAM;
* что и S2+R, и S3+R используют одну и ту же PEFT-конфигурацию.

Достаточно 4–6 предложений, без математики. 

### CLM

Нужно объяснить **на первом упоминании** и везде дальше использовать одну форму:
**causal language modeling (CLM) continued pretraining**.
Не CML.

Достаточно сказать:

* адаптер обучается на сыром корпусном тексте;
* objective — next-token prediction;
* QA labels не используются.

Этого достаточно. 

### RAFT

Тоже объяснить обязательно, но в контексте **твоего** датасета:

* input = question + gold chunks + distractors;
* output = answer;
* это retrieval-conditioned supervision.

Это важнее, чем общее “что такое RAFT в литературе”, потому что у тебя есть конкретная frozen training format. 

### D2L

**Не трать много места.**
Если D2L остаётся только как legacy diagnostic, то ему достаточно:

* 1 абзаца в related attempts / controls / limitations;
* 1–2 предложений про chunk-level workaround;
* 1 строки в таблице результатов.

Полноценное объяснение D2L-механизма в main text не окупается по page budget. Если очень хочется, вынеси в appendix.

### Dualhead LoRA

Я бы **вообще убрал** из тела работы.
Если только не нужно формально показать эволюцию темы — тогда одна сноска в introduction: “The project initially scoped a different adapter comparison, but stabilized around the present experimental design.” Всё.

## 4. Что я бы обязательно подчеркнул в тексте результатов

Вот четыре тезиса, на которых должна стоять вся работа.

**1. Сильный RAG baseline уже очень хорош.**
Это важно, потому что иначе прирост адаптеров трудно интерпретировать. S1 уже даёт `Q_main=0.6425`. 

**2. Адаптеры сверху дают не революцию, а профильный trade-off.**
S2+R и S3+R почти равны по headline quality, но один лучше по deterministic extraction, другой — по free-text quality. Это очень хороший и честный результат.

**3. Retrieval незаменим.**
Переход от S2 к S2+R и от S3 к S3+R огромный; pure parametric controls не заменяют retrieval. Это, пожалуй, самый сильный вывод всей работы.

**4. S7 — отличный, но вторичный результат.**
Он усиливает историю про complementarity, но не должен переписать main claim. Основной claim должен держаться даже если убрать S7 вообще.

## 5. Что делать с графиками и таблицами

### Что точно оставить

**1. Main Results Table**
Оставить обязательно. Это центральный элемент.

Но я бы её немного оформил концептуально:

* разделить строки на `Headline / Post-hoc / Controls`;
* явно пометить S7 как `post-hoc, no retraining`;
* не смешивать narrative: headline comparison — S1, S2+R, S3+R.

**2. Per-Type Score Heatmap**
Это, на мой взгляд, один из лучших твоих графиков. Он идеально показывает, что “кто-то лучше” — слишком грубая формулировка; системы сильны на разных answer types.
Совет: добавь к названиям типов **n** (`boolean n=12`, `date n=5` и т.д.), чтобы читатель не переоценивал маленькие категории.

**3. Judge Criteria Profile**
Тоже хороший график, потому что он помогает объяснить, откуда берётся преимущество S3+R по free-text. Для работы такого объёма он уместен.
Я бы показывал там только **S1, S2+R, S3+R, S7**.

### Что бы я изменил

**Cost quality scatter** в текущем виде я бы либо:

* **переименовал в “Latency–Quality Scatter”**, если по x у тебя latency,
* либо **переделал в настоящий cost–quality plot**, где x = offline cost.

Сейчас название путает: latency — это не cost в том же смысле, что training time / packaging cost. В paper лучше не смешивать это. Если оставляешь текущий график, просто переименуй его. Если делаешь новую версию — лучше использовать `offline cost vs Q_main`, потому что именно это поддерживает practical angle.

### Какой график я бы добавил в первую очередь

**Single-doc vs Multi-doc plot** — обязательно.
У тебя эти цифры уже есть, и это реально сильный результат: multi-doc остаётся больным местом, причём разные системы проседают по-разному. Это намного более содержательно, чем, например, latency-grounding scatter, где grounding у retrieval-aware систем и так одинаковый.

### Какой второй график можно добавить, если есть силы

**Delta vs S1 bar chart**
Для `Q_main`, `S_det`, `S_asst`:

* S2+R vs S1
* S3+R vs S1
* S7 vs S1
* S2 vs S1
* S3 vs S1

Он очень компактно покажет:

* оба retrieval-aware адаптера чуть улучшают baseline;
* controls резко хуже;
* S7 комплементарен.

### Что я бы убрал в appendix

Для 12–20 страниц я бы не тратил main text на:

* latency_grounding_scatter;
* error_overlap_heatmap;
* pairwise_win_heatmap;
* seed_stability;
* pareto_frontier.

Они интересные, но вторичны. Их можно держать в appendix и при этом в основном тексте просто сослаться на них одной фразой. Особенно `latency_grounding_scatter` малоинформативен, если `G` одинаковый у headline retrieval-aware систем.

### Какие таблицы, кроме Main Results, нужны

Минимум ещё одна:

**System Overview Table**
Колонки:

* System
* Retrieval
* Training signal
* Supervision
* Role (headline/control/post-hoc)

Такая таблица сильно экономит текст и сразу делает структуру эксперимента прозрачной.

Опционально — маленькая **Benchmark Summary Table**:

* docs / QA / split / answer types / multi-doc / unanswerable
  Но это можно и в prose, если места мало. 

## 6. Что важно по шаблону оформления

* есть титульный лист;
* есть **Declaration of Academic Integrity**;
* есть оглавление;
* отдельно проговорено, что при использовании GenAI нужно **явно отмечать GenAI-generated passages** и **перечислить инструменты по имени в appendix**, при этом ответственность за них несёт автор. 

То есть практически:

* сделай титульную страницу по шаблону;
* вставь declaration;
* сделай contents;
* добавь в appendix короткий раздел **Use of Generative AI** с перечнем: например, ChatGPT/OpenAI, для чего использовался (обсуждение структуры, редактура формулировок, brainstorming), и в каком объёме;

## 7. Мой итоговый совет в одном абзаце

Я бы писал работу **не как историю про “три способа внедрения знаний”**, а как работу про **ценность parametric adaptation поверх сильного fixed RAG**. Главная ось — **S1 vs S2+R vs S3+R**. Controls нужны, чтобы показать пределы pure parametric memory. S7 нужен, чтобы показать exploratory complementarity, но не как основной claim. Основной вывод должен звучать примерно так: **на компактном legal benchmark сильный RAG уже даёт высокий baseline; parametric adaptation сверху даёт умеренный прирост, но training signal важнее самого факта адаптера; retrieval при этом остаётся незаменимым, а post-hoc merge указывает на частичную комплементарность supervised и supervision-free сигналов.**


---

# Готовая структура работы

## Рабочее название

**Parametric Adaptation over a Strong RAG Baseline for Document-Grounded Legal QA on Consumer Hardware: RAFT-style QLoRA vs. CLM Continued Pretraining**

Если хочешь короче:

**Does Parametric Adaptation Add Value Beyond Strong RAG? A Compact Legal QA Study on Consumer Hardware**

## RQ, которые прямо вставлять в работу

**RQ1.**
Does parametric adaptation add value on top of a strong RAG baseline on a compact legal benchmark under consumer-hardware constraints, and how do RAFT-style supervised adaptation and CLM continued pretraining differ as retrieval-conditioned generators?

**RQ2.**
How far can pure parametric systems go without retrieval on this benchmark, and does retrieval remain indispensable?

## Где объяснять термины

* **RAG, LoRA, QLoRA, CLM, RAFT** — в разделе 2, на первом упоминании, коротко и функционально.
* **Ingestion / indexing / retrieval** — в разделе 3.3, но без глубокого технического экскурса.
* **D2L** — только коротко в разделе 4 как legacy control + отдельная инженерная ремарка в appendix.
* **S6** в основной текст не тащи: он архивирован и не входит в active thesis set. 

## Front matter

### Title page

Что должно быть:

* university / department / subject
* module / seminar / semester / submission date
* thesis title
* lecturer / supervisor
* author details

Это прямо соответствует formal requirements. 

### Declaration of authenticity / academic integrity

Вставь отдельной страницей после титульника.
Если следуешь template буквально, добавь и GenAI statement: что использовалось, как использовалось, и перечисление tools в appendix.  

### Table of contents

Page numbering starts with Introduction as page 1. Title page and contents not counted in the normal numbering. 

---

## 1. Introduction (1.5–2 страницы)

### 1.1 Problem and motivation

**Тезис раздела:**
Modern compact legal QA systems can combine retrieval and parameter-efficient adaptation, but it remains unclear whether parametric adaptation still adds measurable value once the retrieval backbone is already strong.

**Что расписать:**

* legal QA требует factual grounding и устойчивости к lookup-style вопросам;
* на consumer hardware нельзя просто “добросить bigger model”;
* поэтому вопрос не абстрактный, а практический: стоит ли адаптировать генератор, если retrieval уже хорошо сделан?

**Готовые тезисы:**

* “This paper studies whether parametric adaptation still matters when the baseline is not a weak generator-only model, but a strong document-grounded RAG system.”
* “The practical setting is deliberately constrained to consumer hardware, where retrieval engineering and PEFT methods are more realistic than full-scale model training.”

**Вставка:**
Ничего не вставлять. Введение должно быть чистым и читаемым.

---

### 1.2 Research questions and scope

**Тезис раздела:**
The study is intentionally narrow: one compact benchmark, one fixed backbone, one hardware setup, and one frozen retrieval stack.

**Что расписать:**

* 8 DIFC legal documents, 200 QA pairs;
* реальный split: 150 train / 50 eval;
* один backbone: Gemma-2-2b-it;
* одно hardware-ограничение: RTX 4060 8GB;
* сравнение идёт не “вообще всех способов”, а конкретно **на fixed retrieval backbone**.

**Готовые тезисы:**

* “The study is bounded to a compact DIFC legal benchmark with 8 documents and 200 human-authored QA pairs.”
* “All systems are evaluated on the same frozen 50-question evaluation set, while supervised training uses the remaining 150 questions.”
* “This narrow setup is a feature rather than a weakness, because it enables a controlled comparison of training signals under identical infrastructure.”  

**Вставка:**
Ничего не вставлять.

---

### 1.3 Contributions

**Тезис раздела:**
The contribution is not a new architecture, but a controlled empirical comparison with a clean experimental story.

**Что расписать:**

* strong S1 baseline;
* S2+R vs S3+R isolates training signal;
* controls show retrieval necessity;
* S7 suggests complementarity but remains exploratory/post-hoc.

**Готовые тезисы:**

* “The paper contributes a controlled comparison between RAFT-style supervised adaptation and CLM continued pretraining on top of the same RAG backbone.”
* “It also quantifies the limits of pure parametric memory by comparing retrieval-aware systems against no-retrieval controls.”
* “Finally, it reports a post-hoc adapter-merge result that suggests partial complementarity between the two training signals.” 

**Вставка:**
Ничего не вставлять.

---

### 1.4 Structure of the paper

Один короткий абзац:

* Section 2 gives background,
* Sections 3–4 describe benchmark, systems, and evaluation,
* Section 5 presents results,
* Section 6 discusses implications and limitations,
* Section 7 concludes.

Это полезно, потому что official requirements прямо просят **объяснение и justification структуры** во введении. 

---

## 2. Background and related work (1.5–2 страницы)

### 2.1 RAG as nonparametric memory

**Тезис раздела:**
RAG can be viewed as a nonparametric memory mechanism that externalizes document knowledge into the retrieval pipeline instead of forcing the model to memorize it internally.

**Что расписать:**

* very short explanation of RAG;
* почему это релевантно для legal QA;
* почему retrieval особенно важен при compact model + grounded answers.

**Не делать:**

* не расписывать BM25/RRF формулы;
* не уходить в историю RAG на полстраницы.

**Вставка:**
Ничего.

---

### 2.2 Parameter-efficient adaptation on consumer hardware

**Тезис раздела:**
Parameter-efficient adaptation is relevant here not because it is fashionable, but because full fine-tuning is unrealistic under the chosen hardware constraint.

**Что расписать:**

* что такое LoRA;
* что такое QLoRA в 2–4 предложениях;
* почему 4-bit NF4 + adapters адекватны для 8GB VRAM.

**Готовые тезисы:**

* “QLoRA enables controlled adaptation of a small instruction model within consumer-hardware limits.”
* “In this study, PEFT is not an optimization detail but part of the experimental constraint.” 

**Вставка:**
Ничего.

---

### 2.3 RAFT-style adaptation vs. CLM continued pretraining

**Тезис раздела:**
The real comparison in this paper is not ‘adapter vs no adapter’, but ‘supervised retrieval-aware adaptation vs supervision-free corpus adaptation’.

**Что расписать:**

* RAFT-style open-book training: question + gold chunks + distractors → answer;
* CLM continued pretraining: raw corpus text, next-token prediction, no QA labels;
* почему это два разных training signals.

**Готовые тезисы:**

* “RAFT-style adaptation directly optimizes answer generation from evidence-rich contexts.”
* “CLM continued pretraining does not learn the QA task explicitly; it only exposes the adapter to the corpus distribution.”  

**Вставка:**
Ничего.

---

### 2.4 Research gap and positioning

**Тезис раздела:**
Many discussions contrast parametric and nonparametric knowledge at a high level, but fewer studies isolate training signal while holding retrieval and PEFT architecture fixed.

**Что расписать:**

* один абзац собственного positioning;
* почему твоя работа — controlled experimental comparison, а не просто benchmark report.

**Вставка:**
Ничего.

---

## 3. Benchmark and experimental setup (2 страницы)

### 3.1 Corpus and benchmark

**Тезис раздела:**
A compact benchmark is sufficient for a meaningful study if the evaluation setup is carefully controlled and answer types are heterogeneous.

**Что расписать:**

* 8 PDF legal documents, ~115K tokens;
* 200 QA pairs;
* answer types: free_text, boolean, number, name, names, date;
* multi-doc and unanswerable cases;
* реальный split 150/50.

**Готовые тезисы:**

* “The benchmark combines heterogeneous answer types rather than reducing evaluation to free-form summarization.”
* “This matters because it exposes different failure modes, from deterministic extraction to free-text legal explanation.”  

**Вставка:**
Ничего.
Для такого объёма отдельная таблица по датасету не обязательна; лучше сохранить место для результатов.

---

### 3.2 Hardware, shared backbone, and variance policy

**Тезис раздела:**
The study is intentionally bounded to one small backbone and one realistic local hardware configuration.

**Что расписать:**

* Gemma-2-2b-it;
* RTX 4060 8GB, 32GB RAM;
* 3 random seeds for trained systems;
* no cross-validation.

**Готовые тезисы:**

* “The backbone is frozen across all systems to prevent architectural variance from confounding the comparison.”
* “Three seeds are reported for trained systems to capture training variance without exploding experimental cost.” 

**Вставка:**
Ничего.

---

### 3.3 Fixed retrieval backbone

**Тезис раздела:**
The retrieval stack is strong and frozen; this is crucial because it turns S2+R vs S3+R into a comparison of generator adaptation rather than retrieval design.

**Что расписать:**

* ingestion;
* hierarchical chunking;
* dense+sparse hybrid retrieval;
* RRF;
* reranker;
* evidence compression.

**Как глубоко писать:**
Коротко, одной компактной подсекцией. Не “как устроен RAG вообще”, а “что именно зафиксировано в этой работе и почему”.

**Готовые тезисы:**

* “The retrieval backbone is intentionally kept constant across retrieval-aware systems.”
* “As a result, performance differences should be interpreted as differences in how the generator uses retrieved evidence, not as differences in evidence selection itself.”  

**Вставка:**
Ничего.
Отдельная pipeline-схема здесь не обязательна: при 12–20 страницах она съест место.

---

## 4. Compared systems and evaluation protocol (2–2.5 страницы)

### 4.1 System inventory

**Тезис раздела:**
The paper distinguishes clearly between headline systems, controls, and post-hoc results.

**Что расписать:**

* headline: S1, S2+R, S3+R;
* post-hoc: S7;
* controls: S2, S3, S3-legacy (D2L);
* S6 не включать в narrative.

**ВСТАВИТЬ: Table 1 — Compared systems and roles**
Поставить **сразу после 4.1**.

**Что должно быть в Table 1:**

* System
* Retrieval
* Training signal
* Supervision
* Role in paper

Пример строк:

* S1 — yes — none — none — headline baseline
* S2+R — yes — RAFT-style QA — supervised — headline
* S3+R — yes — CLM on corpus — supervision-free — headline
* S7 — yes — merged adapters — post-hoc — post-hoc
* S2 / S3 / S3-legacy — no — control — control

Эта таблица резко упростит чтение и снимет путаницу. 

---

### 4.2 Training setups

**Тезис раздела:**
The central comparison is clean because S2+R and S3+R share the same backbone and PEFT architecture but differ in training signal.

**Что расписать:**

* S2+R: RAFT-style open-book on 150 train questions;
* S3: CLM continued pretraining on concatenated corpus text;
* same PEFT config;
* S7 has no training;
* D2L is legacy and engineering-constrained.

**Готовые тезисы:**

* “The comparison isolates training signal rather than model family.”
* “S7 is reported separately because it is a post-hoc interpolation result, not a retrained system.”
* “D2L is retained only as a historical negative control and not as an equal methodological branch.”  

**Вставка:**
Ничего.

---

### 4.3 Evaluation protocol

**Тезис раздела:**
The evaluation mixes deterministic scoring and judged free-text quality to reflect the heterogeneous nature of the benchmark.

**Что расписать:**

* `Q_main = 0.7 * S_det + 0.3 * S_asst`;
* deterministic scoring for number/boolean/name/names/date;
* judge-based scoring for free_text;
* grounding `G` only for retrieval-aware systems;
* latency, VRAM, offline cost.

**Очень важный тезис:**

* “Because retrieval is fixed, grounding should be interpreted mainly as a property of the shared evidence pipeline rather than of the adapter itself.”
* Можно прямо отметить, что `G=0.5667` у retrieval-aware systems не случайно, а отражает именно этот design choice.  

**Вставка:**
Никакой отдельной фигуры не нужно. Формулу можно дать inline.

---

## 5. Results (4–4.5 страницы)

### 5.1 Main comparison

**Тезис раздела:**
A strong RAG baseline already achieves high quality, so any improvements from adaptation are incremental rather than trivial.

**Что расписать:**

* S1 gives `Q_main=0.6425`;
* S2+R and S3+R both improve it only modestly;
* S7 is highest overall but stays secondary in interpretation because it is post-hoc.

**Готовые тезисы почти в готовом виде:**

* “The strong nonparametric baseline S1 already reaches Q_main=0.6425, making further gains difficult rather than guaranteed.”
* “Both retrieval-aware adapters improve over S1, but the gain is moderate: S2+R reaches 0.6689±0.0137 and S3+R reaches 0.6671±0.0229.”
* “S7 attains the highest score, 0.7045±0.0345, but it is a post-hoc merge result and therefore should not replace the main headline comparison.”  

**ВСТАВИТЬ: Table 2 — Main results table**
Поставить **в начале 5.1**, до текста.

**Как оформить Table 2:**

* сгруппируй строки как Headline / Post-hoc / Control;
* явно подпиши S7 как post-hoc;
* оставь колонки: `Q_main`, `S_det`, `S_asst`, `G`, latency, VRAM, offline cost.

**ВСТАВИТЬ: Figure 1 — Latency–Quality Scatter**
Поставить **сразу после Table 2**, всё ещё в 5.1.

**Важно:**
Переименуй текущий график из **Cost-quality scatter** в **Latency–Quality Scatter**, если по x у тебя latency, а не cost. Offline cost обсуждай текстом по Table 2.

---

### 5.2 Trade-off between S2+R and S3+R

**Тезис раздела:**
The main finding is a trade-off, not a universal winner.

**Что расписать:**

* S2+R better on `S_det`;
* S3+R better on `S_asst`;
* S3+R cheaper offline;
* общий `Q_main` практически tie.

**Готовые тезисы:**

* “The headline comparison does not produce a single dominant practical winner.”
* “S2+R is stronger on deterministic extraction (S_det=0.6479 vs 0.5991), whereas S3+R is stronger on free-text answer quality (S_asst=0.8256 vs 0.7179).”
* “S3+R is also cheaper offline than S2+R, which matters under consumer-hardware constraints.”  

**ВСТАВИТЬ: Figure 2 — Judge Criteria Profile**
Поставить **внутри 5.2**, после первого абзаца.

**Как лучше показать:**

* только S1, S2+R, S3+R, S7;
* без controls, чтобы не захламлять;
* акцент на том, что S3+R выигрывает не “вообще”, а именно в free-text profile.

---

### 5.3 By answer type

**Тезис раздела:**
Different systems have different type-level strengths, which would be hidden by a single aggregate score.

**Что расписать:**

* boolean/number/name/date/names/free_text;
* S2+R vs S3+R разъезжаются по типам;
* date и names остаются проблемными;
* не надо обсуждать каждую ячейку, только patterns.

**Готовые тезисы:**

* “Type-level evaluation shows that the systems do not differ uniformly.”
* “The strongest divergences appear between deterministic extraction and free-text explanation.”
* “Date and multi-name extraction remain weak spots across systems, suggesting persistent formatting and retrieval-use limitations.” 

**ВСТАВИТЬ: Figure 3 — Per-Type Score Heatmap**
Поставить **в 5.3**.

**Совет по оформлению:**

* лучше показывать **S1, S2+R, S3+R, S7**;
* controls можно либо убрать, либо вынести в appendix;
* к названиям типов добавь `n`, например `date (n=5)` и `free_text (n=13)`.

---

### 5.4 Retrieval contribution and the limits of pure parametric memory

**Тезис раздела:**
Retrieval remains indispensable; pure parametric controls fail to replace it.

**Что расписать:**

* S2 vs S2+R;
* S3 vs S3+R;
* S3-legacy D2L as historical negative control;
* это один из главных выводов, а не второстепенная ремарка.

**Готовые тезисы:**

* “Removing retrieval causes a large quality collapse for both training paradigms.”
* “For RAFT, the jump from S2 to S2+R is +0.4059 Q_main; for CLM, the jump from S3 to S3+R is +0.4817.”
* “This indicates that retrieval remains the dominant memory mechanism in this setup, while pure parametric adaptation is insufficient.”  

**Вставка:**
Ничего дополнительного не нужно. Table 2 уже всё показывает.

---

### 5.5 Multi-document difficulty and exploratory adapter fusion

**Тезис раздела:**
Multi-document reasoning remains the hardest regime, and the post-hoc merged adapter suggests only partial—not complete—relief.

**Что расписать:**

* single-doc vs multi-doc comparison;
* S2+R is less bad than S1 and S3+R on multi-doc;
* S7 looks strongest on multi-doc, but interpret carefully because it is post-hoc.

**Готовые тезисы:**

* “Multi-document questions remain substantially harder than single-document questions for nearly all systems.”
* “S1 and S3+R both drop to 0.3100 on multi-doc questions, while S2+R reaches 0.4367 and S7 reaches 0.5233.”
* “This suggests that complementarity between supervised and supervision-free signals may be most useful in the hardest retrieval-conditioned regime, though this remains exploratory.” 

**ВСТАВИТЬ: Figure 4 — Single-doc vs Multi-doc plot (новый график)**
Поставить **в конце 5.5**.

**Какой именно график строить:**

* paired bar chart или dumbbell/lollipop;
* по x — systems;
* по y — `Q_main`;
* две серии: single-doc и multi-doc;
* рядом можно подписать `Δ(multi - single)`.

Это тот новый график, который я бы добавил в первую очередь.

---

## 6. Discussion and limitations (2–2.5 страницы)

### 6.1 Answer to RQ1

**Тезис раздела:**
Parametric adaptation does add value on top of strong RAG, but the improvement is modest and profile-dependent rather than absolute.

**Что расписать:**

* strong baseline already high;
* both S2+R and S3+R improve;
* training signal matters more than the mere presence of an adapter.

**Готовые тезисы:**

* “RQ1 receives a qualified yes: parametric adaptation adds value on top of strong RAG, but the gain is moderate rather than transformative.”
* “More importantly, the choice of training signal changes the generator’s quality profile: RAFT improves deterministic extraction, while CLM improves free-text answer quality.”  

---

### 6.2 Answer to RQ2

**Тезис раздела:**
Pure parametric systems do not come close to replacing retrieval on this benchmark.

**Что расписать:**

* S2 and S3 poor;
* D2L also weak;
* retrieval not optional.

**Готовые тезисы:**

* “RQ2 receives a clear answer: retrieval remains indispensable in this setting.”
* “Neither supervised closed-book adaptation nor corpus-level CLM pretraining provides a viable substitute for external evidence retrieval.” 

---

### 6.3 Error analysis

**Тезис раздела:**
The remaining errors are structured rather than random, which makes them analytically valuable.

**Что расписать:**

* 15 questions all headline systems miss;
* recurring failures: dates, long lists, cross-document composition, unanswerable calibration;
* local complementarity exists, but limited.

**Готовые тезисы:**

* “Error overlap is substantial but not total: 15 questions are missed by all headline systems.”
* “Persistent failure modes include date extraction, long-list normalization, and cross-document composition.”
* “Local wins by S1 and S3+R suggest partial complementarity, but not enough to overturn the main trade-off.” 

**Вставка:**
В main text ничего не вставлять.
Сошлись на appendix figure, если захочешь оставить error-overlap heatmap.

---

### 6.4 Limitations

**Тезис раздела:**
The conclusions are meaningful but deliberately bounded.

**Что расписать:**

* compact corpus;
* 50-question eval;
* one 2B backbone;
* one fixed retrieval stack;
* judge-based free-text scoring;
* S7 is post-hoc;
* D2L used an engineering workaround and is only a legacy control.

**Готовые тезисы:**

* “These findings are bounded to one compact legal corpus, one evaluation split, one backbone, and one consumer-hardware setup.”
* “Because the retrieval backbone is fixed, the study primarily measures differences in evidence-conditioned generation rather than differences in retrieval quality.”
* “The D2L branch is included only as a legacy negative control and should not be overinterpreted as a competitive method in this setup.”  

---

## 7. Conclusion (0.5–1 страница)

**Тезис раздела:**
The paper’s main contribution is a controlled answer to a practical question: whether small-model parametric adaptation still helps once RAG is already strong.

**Что расписать:**

* 3–4 итоговых вывода;
* one practical takeaway;
* future work.

**Готовые тезисы:**

* “A strong compact RAG baseline already delivers high quality on this benchmark.”
* “Parametric adaptation on top of it is beneficial, but the gain is modest and strongly depends on training signal.”
* “Retrieval remains non-substitutable under the studied constraints.”
* “Future work should target multi-document reasoning, unanswerable calibration, and more retrieval-aware adaptation strategies.”

**Вставка:**
Ничего.

---

## Bibliography

По formal requirements:

* только реально процитированная литература;
* alphabetical order;
* без “лишних” источников. 

---

## Appendix

### Appendix A — Hyperparameters and prompts

Что положить:

* QLoRA config;
* CLM training settings;
* judge prompt;
* scoring details.

### Appendix B — Extra tables and figures

Что положить:

* **Table A1 — Full per-type numeric breakdown**
* **Figure A1 — Error overlap heatmap**
* **Figure A2 — Seed stability**
* **Figure A3 — Pairwise win heatmap**
  Это соответствует идее appendix как места для extra statistics / tables. 

### Appendix C — D2L engineering note

Что положить:

* короткое объяснение, что D2L в реальном прогоне потребовал chunk-level workaround;
* почему это legacy control;
* почему в основной narrative он вторичен.
  Это поможет снять вопрос о несостыковке документации.

### Appendix D — Use of Generative AI

Если следуешь template строго:

* перечисли tool names;
* для чего использовались;
* при необходимости отметь passages according to supervisor policy. 

---

# Где что вставлять: краткая карта

### В основном тексте

* **Table 1 — Compared systems and roles** → после **4.1**
* **Table 2 — Main results table** → в начале **5.1**
* **Figure 1 — Latency–Quality Scatter** → после Table 2, в **5.1**
* **Figure 2 — Judge Criteria Profile** → в **5.2**
* **Figure 3 — Per-Type Score Heatmap** → в **5.3**
* **Figure 4 — Single-doc vs Multi-doc plot** → в **5.5**

### В appendix

* **Table A1 — Full per-type numeric breakdown**
* **Figure A1 — Error overlap heatmap**
* **Figure A2 — Seed stability**
* **Figure A3 — Pairwise win heatmap**

---

# Приоритет визуализаций, если начнёт не хватать места

Оставить обязательно:

1. **Table 1**
2. **Table 2**
3. **Figure 3**
4. **Figure 4**

Оставить желательно:
5. **Figure 2**
6. **Figure 1**

Убрать первым делом:

* error overlap heatmap из main text,
* seed stability,
* pairwise wins,
* pareto/frontier.

---

# Самая короткая версия narrative, которую стоит держать в голове при письме

**Сильный fixed RAG уже даёт высокий baseline. Parametric adaptation поверх него действительно помогает, но training signal важнее самого факта адаптера: RAFT сильнее в deterministic extraction, CLM — в free-text answer quality и offline efficiency. Retrieval при этом остаётся незаменимым, а post-hoc merge указывает на частичную комплементарность сигналов, не отменяя основной headline comparison.**  
