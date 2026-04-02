# Литература по параметрической адаптации и RAG на потребительском железе

## 1. Базовые работы по LoRA и QLoRA

1. **Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models” (ICLR 2022).** Вводит LoRA как способ заморозить базовую модель и обучать только низкоранговые адаптеры, резко уменьшая число обучаемых параметров и память без потери качества.[^1]
2. **Dettmers et al., “QLoRA: Efficient Finetuning of Quantized LLMs” (arXiv 2305.14314).** Показывает, как за счёт NF4‑квантования, double quantization и paged‑оптимизаторов обучать LoRA‑адаптеры поверх 4‑битовой копии модели, достигая качества полноточной дообученной модели при сильной экономии VRAM.[^2][^3][^4]
3. **Hugging Face blog, “Making LLMs even more accessible with bitsandbytes, 4‑bit …”** Практическое описание QLoRA‑pipeline c 4‑битовым quantization в `bitsandbytes` и PEFT, с инженерным фокусом на доступность больших моделей на одном GPU.[^4]
4. **McCormick, “QLoRA and 4‑bit Quantization” (2024, технический разбор).** Подробный туториал по NF4, схеме сжатия/десжатия и практическим аспектам QLoRA, полезен для раздела о реализации на RTX 4060.[^5][^6]
5. **“Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey” (arXiv 2403.14608).** Обзор PEFT‑методов (LoRA, Prefix‑tuning, BitFit и др.) с акцентом на компромисс «качество/ресурсы», системные аспекты и сценарии использования.[^7][^8]
6. **“Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment” (arXiv 2312.12148).** Более ранний, но полезный обзор PEFT для PLM/LLM; можно использовать в разделе «связанная работа» как обобщение поля.[^9]

## 2. Doc-to-LoRA и контекст→адаптер (Doc-to-LoRA / D2L)

1. **“Doc-to-LoRA: Learning to Instantly Internalize Contexts” (arXiv 2602.15902).** Предлагает D2L‑гиперсеть, которая по длинному контексту генерирует LoRA‑адаптер для целевой LLM, позволяя последующим запросам работать без повторного потребления исходного документа и уменьшая KV‑cache и латентность.[^10][^11]
2. **Обсуждения/страница статьи на Hugging Face “Doc-to-LoRA: Learning to Instantly Internalize Contexts”.** Даёт краткий обзор и подчёркивает, что D2L рассматривается как приближённый context distillation с упором на быстрые обновления и персонализацию.[^12][^13]
3. **Новостные и обзорные заметки по D2L (например, AI‑Navigate, ChatPaper).** Полезны как дополнительные источники при формулировке вашего раздела о D2L как негативном контроле и мотиве для CLM‑ветки.[^14][^13]

## 3. RAFT и обучение генератора в режиме open‑book

1. **Zhang et al., “RAFT: Adapting Language Model to Domain Specific RAG” (arXiv 2403.10131).** Вводит Retrieval‑Augmented Fine‑Tuning (RAFT): обучающие примеры состоят из вопроса, множества retrieved документов (включая дистракторы) и chain‑of‑thought ответа; модель учится игнорировать нерелевантные документы и цитировать релевантные.[^15][^16][^17]
2. **Сводные обзоры RAFT (например, SummarizePaper).** Уточняют, что RAFT рассматривается как пост‑тренировочный рецепт для улучшения in‑domain RAG и показывает выгоду по сравнению с чистым prompting.[^18][^19]
3. **Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks” (RAG, 2020).** Классическая работа по совмещению параметрической и непараметрической памяти (seq2seq + dense retriever), задаёт каноническую формулировку RAG и типовую архитектуру, на фоне которой вы позиционируете свой S1.[^20][^21][^22]

## 4. Обзоры и варианты PEFT/LoRA для ограниченного железа

1. **LoRA‑обзоры и заметки (“A Note on LoRA”, LoRA+, LoRA‑FA, LoTR и др.).** Эти работы расширяют базовый LoRA (отдельные LR/η для A/B, улучшенные инициализации, тензорные обобщения) и дают материал для обсуждения выбора конфигурации адаптера и ранга в малой модели.[^23][^24][^25][^26]
2. **“Parameter-Efficient Fine-Tuning in Large Models: A Survey of …” (arXiv 2410.19878).** Поздний обзор PEFT с акцентом на ограничения по памяти/вычислениям и практические системы, релевантен вашей мотивации «потребительское железо, 8 ГБ VRAM».[^27]
3. **Empirical evaluation of low-rank adaptation for efficient fine-tuning of large language models (Turing Technical Report).** Эмпирически исследует LoRA на небольших моделях, фокусируясь на памяти и времени обучения; может лечь в раздел о практических trade‑off’ах.[^28]

## 5. Мерджинг и композиция адаптеров (для S7 и обсуждения линейной интерполяции)

1. **“Merging LoRAs for Practical Skill Composition Tasks” (CAT, 2024).** Исследует различные схемы линейного слияния двух LoRA‑адаптеров (конкатенация, DARE, majority‑sign и др.) и показывает, что аккуратный merge может приблизиться к multi‑task обучению без доп. тренировки.[^29]
2. **“Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering” (LoRA‑LEGO, 2024).** Предлагает rank‑wise кластеризацию компонент LoRA и построение объединённого адаптера по центроидам; полезно как более продвинутый фон для вашего простого линейного merge (S7).[^30][^31]
3. **“Rethinking Inter-LoRA Orthogonality in Adapter Merging” (2025).** Анализирует идею ортогональности между LoRA‑модулями и показывает, что одна лишь ортогональность не гарантирует семантическую композиционность; важно для обсуждения ограничений вашего S7‑merge.[^32]
4. **“Task-Aware LoRA Adapter Composition via Similarity Retrieval in Vector Databases” (2026).** Использует retrieval по embedding’ам задач для динамического выбора и взвешенного слияния нескольких LoRA‑адаптеров; полезен как пример более сложной, но родственной идеи «адаптер‑как‑модуль», в отличие от статичного S7.[^33]
5. **Работы по онлайн‑мерджингу и continual LoRA (K‑Merge и др.).** Исследуют сценарий последовательного слияния LoRA под жёстким бюджетом хранения, что можно упомянуть в разделе о перспективах частых обновлений юридического корпуса на потребительском железе.[^34][^35][^36]

## 6. RAG и оценка в юридическом домене

1. **LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models.** Предлагает 162 задач по юридическому рассуждению; даёт хороший фон для раздела «связанные бенчмарки», хотя ваш корпус компактнее и более прикладной.[^37][^38][^39]
2. **LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain.** Фокусируется на оценке именно retrieval‑компоненты RAG для юр. текстов и вводит облегчённый LegalBench‑RAG‑mini, что созвучно вашему компактному DIFC‑корпусу.[^40]
3. **LRAGE: Holistic Evaluation of RAG Systems in the Legal Domain.** Предлагает инструмент и методологию для факторного анализа влияния частей RAG‑пайплайна (корпус, retriever, reranker, LLM, метрики) на финальную точность; релевантно вашему разбору вклада retrieval против адаптеров.[^41]
4. **Legal RAG Bench: an end-to-end benchmark for legal RAG.** Ещё один свежий бенчмарк для юридического RAG, с акцентом на end‑to‑end оценку и иерархическую декомпозицию ошибок; полезен как внешний референс к вашему Q_main и разбиениям по типам ответов.[^42]
5. **LLM-as-a-Judge: Rapid Evaluation of Legal Document …** Описывает использование LLM‑судьи и статистических тестов для оценки legal‑RAG‑систем; методологически близко к вашему использованию LLM‑judge и агрегированных метрик.[^43][^44]

## 7. Общие обзоры и системы для RAG

1. **“Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks” (RAG).** Ключевая работа, описывающая архитектуру RAG и показывающая выигрыш над чисто параметрическими моделями; задаёт теоретическую мотивацию для совмещения retrieval и адаптаций.[^21][^45][^22]
2. **Работы по LegalBench-RAG и Legal RAG Bench (см. выше).** Они дополняют общий RAG‑фон юридически специфичными постановками задач и метриками.[^40][^42]

## 8. Дополнительные источники для разделов «Связанная работа» и «Обсуждение ограничений»

1. **Современные варианты QLoRA и квантования (LoftQ, IR‑QLoRA, QUAD, LLM‑FP4).** Эти работы улучшают качество под 2–4‑битным квантованием и показывают, как минимизировать деградацию при агрессивной компрессии; хороши для дискуссии о возможных расширениях за пределы базового QLoRA.[^46][^47][^48][^49]
2. **Системные работы типа S‑LoRA (Serving Thousands of Concurrent LoRA Adapters).** Показывают, как масштабно сервить множество LoRA‑адаптеров, что можно упомянуть как более «серверный» контраст к вашему single‑GPU, single‑adapter сценарию.[^50]
3. **Работы по юридическим RAG‑системам и их статистической оценке (EARL, рекомендации по метрикам и тестам).** Полезны для раздела о статистической значимости разницы между системами и выборе тестов/коррекций в юридическом контексте.[^44][^43]

***

Эти блоки можно напрямую развернуть в раздел «Related Work» вашей работы, группируя источники по измеряемым вами осям: (1) PEFT/QLoRA под ограниченное железо, (2) методы контекст→адаптер (Doc‑to‑LoRA, context distillation), (3) RAFT‑подобное open‑book обучение генератора, (4) мерджинг адаптеров и композиция навыков, (5) RAG и его оценка в юридическом домене.

---

## References

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - An important paradigm of natural language processing consists of large-scale pre-training on general...

2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65...

3. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf) - We present QLoRA, an efficient finetuning approach that reduces memory usage
enough to finetune a 65...

4. [Making LLMs even more accessible with bitsandbytes, 4-bit ...](https://huggingface.co/blog/4bit-transformers-bitsandbytes) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

5. [QLoRA and 4-bit Quantization](http://mccormickml.com/2024/09/14/qlora-and-4bit-quantization/) - An in-depth tutorial on the algorithm and paper, including a pseudo-implementation in Python.

6. [S4. Natural Float4 (“NF4”)](https://mccormickml.com/2024/09/14/qlora-and-4bit-quantization/) - An in-depth tutorial on the algorithm and paper, including a pseudo-implementation in Python.

7. [Parameter-Efficient Fine-Tuning for Large Models - arXiv](https://arxiv.org/html/2403.14608v3) - In this survey, we present comprehensive studies of various PEFT algorithms, examining their perform...

8. [Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey](https://arxiv.org/abs/2403.14608) - Large models represent a groundbreaking advancement in multiple application fields, enabling remarka...

9. [Parameter-Efficient Fine-Tuning Methods for Pretrained Language ...](https://ai.updf.com/paper-detail/parameter-efficient-fine-tuning-methods-for-pretrained-language-models-a-xu-xie-83de13bd492b9e72c314e308f0d77014154a6a74) - This survey serves as an invaluable resource for researchers and practitioners seeking to navigate t...

10. [Doc-to-LoRA: Learning to Instantly Internalize Contexts - arXiv.org](https://www.arxiv.org/abs/2602.15902) - Long input sequences are central to in-context learning, document understanding, and multi-step reas...

11. [Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://arxiv.org/abs/2602.15902v1) - Long input sequences are central to in-context learning, document understanding, and multi-step reas...

12. [Paper page - Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://huggingface.co/papers/2602.15902) - Join the discussion on this paper page

13. [Doc-to-LoRA: Learning to Instantly Internalize Contexts - ChatPaper](https://chatpaper.com/paper/238644) - Doc-to-LoRA (D2L) is a novel method that enhances the efficiency of long input sequence processing i...

14. [[R] Doc-to-LoRA: Learning to Instantly Internalize Contexts from ...](https://ai-navigate-news.com/en/articles/c0eb8e34-e829-43e2-af46-b8be9e214a83) - - Doc-to-LoRA introduces a lightweight hypernetwork that meta-learns to generate LoRA adapters for a...

15. [[PDF] arXiv:2403.10131v2 [cs.CL] 5 Jun 2024](https://arxiv.org/pdf/2403.10131.pdf) - In this paper, we present Retrieval Augmented. Fine Tuning (RAFT), a training recipe which improves ...

16. [RAFT: Adapting Language Model to Domain Specific RAG - arXiv](https://arxiv.org/html/2403.10131v1) - In this paper, we present Retrieval Augmented Fine Tuning (RAFT), a training recipe that improves th...

17. [RAFT: Adapting Language Model to Domain Specific RAG - arXiv](https://arxiv.org/abs/2403.10131) - In this paper, we present Retrieval Augmented FineTuning (RAFT), a training recipe that improves the...

18. [Results of the summarizing process for the arXiv paper: 2403.10131v1](https://summarizepaper.com/en/arxiv-id/2403.10131v1/) - Easy-to-read summary of the arXiv paper 2403.10131v1 entitled RAFT: Adapting Language Model to Domai...

19. [Results of the summarizing process for the arXiv paper: 2403.10131v2](https://www.summarizepaper.com/en/arxiv-id/2403.10131v2/) - Easy-to-read summary of the arXiv paper 2403.10131v2 entitled RAFT: Adapting Language Model to Domai...

20. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://tldr.takara.ai/p/2005.11401) - We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models w...

21. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Large pre-trained language models have been shown to store factual knowledge in their parameters, an...

22. [Retrieval-augmented generation for knowledge-intensive NLP tasks](https://dl.acm.org/doi/abs/10.5555/3495724.3496517) - We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models wh...

23. [LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models
  Fine-tuning](https://arxiv.org/pdf/2308.03303.pdf) - The low-rank adaptation (LoRA) method can largely reduce the amount of
trainable parameters for fine...

24. [LoTR: Low Tensor Rank Weight Adaptation](http://arxiv.org/pdf/2402.01376.pdf) - In this paper we generalize and extend an idea of low-rank adaptation (LoRA)
of large language model...

25. [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/pdf/2402.12354.pdf) - In this paper, we show that Low Rank Adaptation (LoRA) as originally
introduced in Hu et al. (2021) ...

26. [A Note on LoRA](http://arxiv.org/pdf/2404.05086.pdf) - LoRA (Low-Rank Adaptation) has emerged as a preferred method for efficiently
adapting Large Language...

27. [Parameter-Efficient Fine-Tuning in Large Models: A Survey of ... - arXiv](https://arxiv.org/abs/2410.19878) - The large models, as predicted by scaling raw forecasts, have made groundbreaking progress in many f...

28. [Empirical evaluation of low-rank adaptation for efficient fine ...](https://zenodo.org/records/16417805) - This study evaluates Low-Rank Adaptation (LoRA) as a parameter-eficient method for fine-tuning large...

29. [Merging LoRAs for Practical Skill Composition Tasks - arXiv](https://arxiv.org/html/2410.13025v2) - This paper advocates model merging as an efficient way to solve compositional tasks and underscores ...

30. [Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to
  Extremes Through Rank-Wise Clustering](https://arxiv.org/pdf/2409.16167.pdf) - ...Building on these insights,
we propose the LoRA-LEGO framework. This framework conducts rank-wise...

31. [Merging LoRAs like Playing LEGO: Pushing the Modularity of ... - arXiv](https://arxiv.org/html/2409.16167v1) - This paper approaches LoRA merging from a novel perspective, focusing on the fine-grained modulariza...

32. [[2510.03262] Rethinking Inter-LoRA Orthogonality in Adapter Merging](https://arxiv.org/abs/2510.03262) - Abstract page for arXiv paper 2510.03262: Rethinking Inter-LoRA Orthogonality in Adapter Merging: In...

33. [Task-Aware LoRA Adapter Composition via Similarity Retrieval in Vector Databases](https://arxiv.org/abs/2602.21222) - Parameter efficient fine tuning methods like LoRA have enabled task specific adaptation of large lan...

34. [Towards Reversible Model Merging For Low-rank Weights](https://arxiv.org/abs/2510.14163) - Model merging aims to combine multiple fine-tuned models into a single set of weights that performs ...

35. [K-Merge: Online Continual Merging of Adapters for On-device Large ...](https://arxiv.org/html/2510.13537v1)

36. [Tensorized Clustered LoRA Merging for Multi-Task Interference - arXiv](https://arxiv.org/abs/2508.03999) - Abstract:Despite the success of the monolithic dense paradigm of large language models (LLMs), the L...

37. [LegalBench: A Collaboratively Built Benchmark for ...](https://huggingface.co/papers/2308.11462) - Join the discussion on this paper page

38. [LegalBench: A Collaboratively Built Benchmark for Measuring Legal ...](https://ar5iv.labs.arxiv.org/html/2308.11462) - To enable greater study of this question, we present LegalBench: a collaboratively constructed legal...

39. [LegalBench: A Collaboratively Built Benchmark for ...](https://arxiv.org/abs/2308.11462) - The advent of large language models (LLMs) and their adoption by the legal community has given rise ...

40. [LegalBench-RAG: A Benchmark for Retrieval-Augmented ... - arXiv.org](https://arxiv.org/html/2408.10343v1)

41. [2504.01840](https://papers.cool/arxiv/2504.01840) - Recently, building retrieval-augmented generation (RAG) systems to enhance the capability of large l...

42. [Legal RAG Bench: an end-to-end benchmark for legal RAG](https://deeplearn.org/arxiv/708587/legal-rag-bench:-an-end-to-end-benchmark-for-legal-rag) - Things happening in deep learning: arxiv, twitter, reddit

43. [LLM-as-a-Judge: Rapid Evaluation of Legal Document ... - arXiv.org](https://arxiv.org/html/2509.12382v1)

44. [How to evaluate legal RAG with statistical tools - LinkedIn](https://www.linkedin.com/posts/madhavan-seshadri_our-earl-recsys-25-paper-tackles-evaluation-activity-7374425696166789121-5cxS) - Our EARL @ RecSys '25 paper tackles evaluation bottlenecks in legal RAG by revisiting classic statis...

45. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://huggingface.co/papers/2005.11401) - Join the discussion on this paper page

46. [Accurate LoRA-Finetuning Quantization of LLMs via Information Retention](https://arxiv.org/pdf/2402.05445.pdf) - ...the perspective of unified information: (1)
statistics-based Information Calibration Quantization...

47. [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/pdf/2310.16836.pdf) - We propose LLM-FP4 for quantizing both weights and activations in large
language models (LLMs) down ...

48. [QUAD: Quantization and Parameter-Efficient Tuning of LLM with Activation
  Decomposition](https://arxiv.org/html/2503.19353v1) - ...Llama-3-8B) due to activation outliers. To address this, we propose QUAD
(Quantization with Activ...

49. [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/pdf/2310.08659.pdf) - ...full-precision model and significantly improves generalization in
downstream tasks. We evaluate o...

50. [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/pdf/2311.03285.pdf) - ...pretrain-then-finetune" paradigm is commonly adopted in the deployment
of large language models. ...

