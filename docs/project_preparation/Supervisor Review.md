Есть один важный теоретический момент, который я бы обязательно зафиксировал в тексте работы. Если твой **4-head MH-LoRA** — это просто четыре независимые ветки ((A_i,B_i)), которые **всегда одновременно активны** и их вклад просто суммируется, то  

$\Delta W=\sum_{i=1}^4 B_iA_i=[B_1\ B_2\ B_3\ B_4]  \begin{bmatrix}  A_1\A_2\A_3\A_4  \end{bmatrix}$ 

то есть по классу функций это эквивалент обычной rank-32 LoRA. LoRA в принципе и задаёт обновление как (BA); поэтому твой comparison 2 vs 3 — это не “больше выразительности”, а **другая параметризация и другие optimization dynamics**. Это не проблема, но это надо честно сформулировать. И именно здесь полезна мотивация из R-LoRA: multi-head конструкции без механизма диверсификации часто стремятся к похожим head-матрицам, поэтому если ты хочешь, чтобы пункт 3 был именно исследовательским, а не почти-эквивалентной репараметризацией, добавь хотя бы **random init по головам + head dropout** как лёгкий diversity mechanism.

Ниже — уже готовый каркас, который можно почти переносить в план.

## Рабочая тема

Я бы взял одну из этих формулировок:

**Parameter-Efficient Knowledge Injection for Document-Grounded QA on Consumer Hardware: Classical RAG, Single-Head LoRA, 4-Branch LoRA, and Cluster-Routed LoRA**

или

**Nonparametric, Parametric, and Hybrid Knowledge Injection for Document-Grounded QA on Consumer Hardware: a Legal QA Case Study**

Обе лучше, чем чисто “legal QA” в заголовке: legal QA остаётся практическим полигоном, а не всей сутью работы. Это ещё и хорошо согласуется с твоим exposé, где уже есть consumer hardware, frozen backbone и 4-head setup как исходная ось исследования.

## Research question

**При фиксированном бюджете обучаемых параметров и на потребительском GPU, какая стратегия внедрения знаний в document-grounded QA даёт лучший компромисс между качеством ответа, groundedness и latency: classical RAG, single-head LoRA, 4-branch MH-LoRA, cluster-routed single-head LoRA или гибрид RAG + adapter?**

## Гипотезы

**H1.** Гибридный подход (**RAG + лучший adapter**) даст лучший общий practical trade-off, чем чисто непараметрический и чисто параметрический режимы, потому что retrieval даёт актуальный внешний контекст, а адаптер подстраивает генерацию под домен и стиль ответа. Это хорошо согласуется с RAFT как open-book training recipe и с routed/parametric RAG-идеями вроде Poly-PRAG. (

**H2.** Обычный 4-branch MH-LoRA без механизма диверсификации не даст большого отрыва от single-head LoRA при равном бюджете параметров, потому что это почти та же rank-32 адаптация в другой факторизации; заметный эффект возможен именно как эффект регуляризации/оптимизации. 

**H3.** Cluster-routed single-head LoRA сможет обойти both single-head и plain MH-LoRA, если кластеры действительно соответствуют устойчивым типам запросов/локальным поддоменам; если же кластеризация шумная, routed-вариант будет нестабилен и может проиграть более “гладкому” MH-LoRA. Идея routed adapters хорошо опирается на литературу по adapter routing и latent expert routing. 

**H4.** HyDE должен улучшать прежде всего retrieval-dependent системы — обычный RAG и hybrid RAG+adapter — потому что он меняет retrieval query representation, а не сам generator. Для pure parametric LoRA-систем он просто нерелевантен. 

## Финальный набор систем

Я бы сделал именно такой минимальный, но уже сильный набор:

**1. Classical RAG**  
Твой текущий vector-base pipeline без fine-tuning. Это nonparametric baseline.

**2. Single-head LoRA**  
Одна LoRA, rank 32. Это parametric baseline. Если идёшь на 4060 8GB, QLoRA — естественный training recipe, потому что он как раз сводит память к реалистичному 4-bit режиму для PEFT. 

**3. 4-branch MH-LoRA**  
Четыре независимые ветки rank 8 с собственными (A_i,B_i), суммируемые в один update. Я бы рекомендовал либо явно оставить его как “decomposed rank-32 LoRA”, либо добавить лёгкий diversity mechanism: random init per head + head dropout. Если времени мало, делай одну основную версию и в тексте честно пиши, что это study of decomposition/optimization, а не study of extra expressive power. R-LoRA здесь главный методологический якорь. 

**4. Cluster-routed single-head LoRA**  
Четыре отдельных single-head LoRA-адаптера, каждый обучается на своём кластере, а на inference активируется только один. Чтобы comparison был честным по **общему числу trainable parameters**, я бы сделал **4 адаптера rank 8**, а не 4 адаптера rank 32. Тогда суммарный обучаемый бюджет совпадает с single-head rank 32 и с твоим 4-branch MH-LoRA. Это и будет главный comparison:  
**“implicit specialization inside one decomposed adapter” vs “explicit specialization by routed experts”**.

**5. Hybrid: Classical RAG + best adapter**  
Берёшь лучший из пунктов 2–4 и запускаешь вместе с retrieval. Это практический top-line.

## Почему именно cluster-routed single-head, а не cluster-routed 4-head

Потому что **routed 4-head LoRA смешивает сразу две оси изменения**:

1. внутренняя multi-branch факторизация одного адаптера;
2. внешняя expert routing между несколькими адаптерами.

Для курсовой это слишком грязный дизайн: если routed 4-head победит, непонятно, за счёт чего именно — routing, internal heads, или просто роста архитектурной сложности. Поэтому твоя исходная идея — **сравнить 4-head MH-LoRA с cluster-routed single-head LoRA** — методологически чище и сильнее.

## Протокол эксперимента без синтетических данных

Здесь можно полностью уважить твоё желание **не генерировать synthetic train set**.

Я бы сделал так:

**Основной протокол:**  
5-fold cross-validation на твоих 100 вопросах. В каждом fold:

- 80 train+val
- 20 test  
    А внутри 80 ещё небольшой val-split для выбора LR / epoch / early stopping.

Это важнее, чем один фиксированный split, потому что на 100 вопросах variance будет высокой, а LoRA-методы чувствительны к обучающему режиму и LR. Это подтверждается и в работах по LoRA+, и в обзорах по вариантам LoRA: хорошая настройка optimizer/LR часто меняет картину сильнее, чем “модный” вариант адаптера. 

**Что фиксируешь жёстко для всех систем:**

- один backbone;
- один tokenizer;
- один prompt format;
- один retriever / chunking / top-k / reranking;
- один список target modules для LoRA;
- один parameter budget.
    

**Что я бы рекомендовал по backbone:**  
Судя по твоему exposé, логично сделать **Llama-3.2-1B** основной моделью, а **3B** — только как маленький confirmatory run, если останется время. На 4060 8GB это просто намного безопаснее с точки зрения цикла экспериментов. QLoRA это хорошо поддерживает как базовый рецепт PEFT на квантованной модели. 

**Обучение без синтетики:**  
Для parametric систем 2–4 делай closed-book fine-tuning на train QA pairs.  
Для hybrid-системы 5 можешь оставить тот же адаптер и просто добавить retrieval на inference.

Если захочешь маленькое улучшение **без synthetic generation**, добавь в appendix RAFT-style open-book variant на тех же train вопросах: question + gold chunk(s) + 1–2 distractors → answer. Это не новые сгенерированные данные, а просто другой формат уже имеющихся train examples. RAFT как раз показывает ценность обучения в open-book режиме с distractor documents. 
## Роутер для cluster-routed single-head LoRA

Я бы не усложнял.

**Основной вариант роутера:**

- embedding только вопроса;
- k-means, (k=4), на train fold;
- на inference — nearest centroid по cosine similarity.

Это сильный baseline: он прост, дешёв, объясним и не переобучается так быстро, как learned router на 100 вопросах.

**Почему не learned router сразу:**  
на таком размере данных он легко будет учить шум, а не структуру. Если останется время, learned router можно добавить как appendix: маленький MLP на question embedding с pseudo-labels “какой адаптер дал лучший val loss”. Но в core design я бы этого не делал.

## Метрики: что оставить, что адаптировать

Здесь я бы сделал **двухслойную оценку**.

### 1) Главная метрика для сравнения всех систем

$Q = 0.7S_{det} + 0.3S_{asst}$  

где:

- ($S_{det}$) — accuracy по deterministic questions,
- ($S_{asst}$) — LLM-judge по free-text.

Это берёт лучшее из challenge-метрики и остаётся **применимым ко всем системам**, включая pure parametric. В твоём challenge file именно эти два компонента лежат в основе total score.

### 2) Retrieval-aware слой только для RAG / Hybrid

Для систем, где retrieval есть, отдельно считаешь:

$G = F_{\beta}, \quad \beta = 2.5$  

на page-level grounding, как в challenge. Это хорошая идея, потому что в исходной системе оценки grounding — самый важный множитель, причём recall там приоритизируется. Для RAG и Hybrid это очень уместно. Для pure parametric систем я бы **не делал вид, что этот G можно честно считать тем же способом**; для них он должен быть `N/A`, а не ноль.

### 3) Latency / systems metrics

Отдельно и обязательно:

- **TTFT**
- **end-to-end latency**
- **peak VRAM**
- **training time per epoch / total wall-clock**

В challenge TTFT действительно влияет на итоговый multiplier, но для курсовой я бы не встраивал его multiplicatively в главный academic score. Лучше показывать его отдельно и, при желании, как auxiliary factor для retrieval systems.

### 4) Что я бы убрал из headline metrics

**Telemetry factor (T)** я бы не делал центральной метрикой работы. Для челленджа он важен как проверка корректности submission format, но для исследовательской курсовой это скорее sanity check реализации, а не свойство модели. Его можно упомянуть в разделе evaluation pipeline, но не тащить в главный результат.

### 5) Какой breakdown обязательно показать

Результаты показывай не только aggregate, но и по типам вопросов:

- number
- boolean
- name / names
- date
- free_text
- unanswerable / null

Это прямо следует из challenge formulation и сильно повысит интерпретируемость.

## Два самых перспективных анализа “специализации”

Ты просил не раздувать список, поэтому я бы оставил только эти два.

**1. Heatmap “head / adapter usage vs question type”**  
Для 4-head MH-LoRA смотри вклад каждой головы по норме её update/output на вопросах разных типов.  
Для cluster-routed single-head LoRA — какой адаптер выбирается чаще для каких типов вопросов.

Если увидишь устойчивый рисунок вроде “adapter 3 чаще работает на date/number, adapter 1 — на clause analysis, adapter 4 — на unanswerable”, это уже сильный interpretability result.

**2. Pairwise similarity matrix между головами / адаптерами**  
Считай cosine similarity между flattened (\Delta W_i) или между head outputs на фиксированном probe-set.  
Это очень полезно, потому что сразу отвечает на вопрос: головы реально разошлись или collapse-нулись в почти одну и ту же подзадачу. Именно проблема слишком похожих head matrices явно обсуждается в R-LoRA. 

Этих двух графиков тебе достаточно, чтобы добавить в работу не просто benchmarking, а нормальную интерпретационную часть.

## Минимальный набор литературы, который я бы оставил “обязательным”

Если сильно сократить список, я бы оставил ядро из восьми работ:

- **LoRA**
- **QLoRA**
- **HyDE**
- **RAFT**
- **MELoRA**
- **R-LoRA**
- **Multi-Head Adapter Routing for Cross-Task Generalization**
- **Poly-PRAG**

Это уже даёт тебе: базовый PEFT, low-memory training, retrieval-side augmentation, open-book adaptation, parallel mini-LoRA/multi-branch intuition, head diversification, routing literature и parametric-routed RAG bridge. 
Если собрать всё в одну очень короткую формулу, то твоя работа теперь звучит так:

**Ты сравниваешь nonparametric, parametric и hybrid knowledge injection на фиксированном корпусе и ограниченном железе, а внутри parametric ветки отдельно проверяешь: лучше ли implicit decomposition одного rank-32 adapter в 4 ветки, чем explicit routing между 4 маленькими single-head adapters.**

Это уже очень цельный и хорошо защищаемый дизайн.

Следующим сообщением я могу превратить этот каркас в **готовый академический текст**: цель работы, задачи, объект/предмет, гипотезы и раздел “Методология” в стиле, который можно почти вставить в документ.