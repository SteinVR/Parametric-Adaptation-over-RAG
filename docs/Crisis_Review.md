В текущем виде проект **не бессмысленный**, но **Architecture v7 пытается рассказывать две разные истории одновременно**, и из-за этого всё выглядит “жидко”. С одной стороны, SSOT уже честно признаёт, что `S2` как **closed-book QLoRA** — это чистый parametric stress test, а не ожидаемый победитель: там прямо зафиксирована гипотеза, что 2B-модель не запомнит 176 страниц по 150 train-QA и, вероятно, проиграет `S1`. С другой стороны, фактический положительный результат у тебя сейчас уже есть не по Axis 1, а по Axis 2: **`S2+R` (RAFT + retrieval) уже beat’ит `S1`** с `Q_main=0.669±0.014` против `0.6425`, то есть реальный сигнал пока идёт именно из retrieval-augmented ветки.  

Иными словами, **да, “QLoRA without retrieval” как главный герой выглядит безнадёжно**, но отсюда не следует, что вся работа развалилась. Это значит только одно:
**первоначальный question про “парадигмы в изоляции” больше не должен быть главным narrative.**
Он остаётся полезным как **falsification / limits axis**: можно проверить, насколько далеко вообще удаётся уйти от retrieval в компактном legal benchmark. Но делать вид, что это симметричный horse race между `S1`, `S2`, `S3`, `S4`, уже не стоит. Это даже твой текущий SSOT фактически признаёт: Axis 1 — “paradigm in isolation”, Axis 2 — “retrieval augmentation”, и при этом именно Axis 2 уже даёт meaningful win.  

При этом сам benchmark ты уже **сильно починил**. Сейчас это не тот старый разреженный ад на 65 документах: у тебя **8 PDF, 176 страниц, ~115K токенов**, и главное — **каждый документ помещается в один D2L single pass**. Это очень важная победа: ты убрал главную инженерную катастрофу старой постановки, где весь смысл тонул в merge-of-40. На таком корпусе уже можно честно спрашивать, что даёт retrieval, что даёт supervised adapter, и что даёт hypernetwork packaging. 

Так что мой ответ на твой первый вопрос такой:

## 1) “А к чему это всё вообще было? Актуален ли исходный вопрос?”

**Да, но не в той форме, в какой он сейчас стоит в SSOT.**
Если держаться за формулу “nonparametric vs supervised parametric vs hypernetwork parametric in isolation”, то история становится искусственной, потому что `S2` почти по определению проигрывает и нужен не как contender, а как **negative control**. Тогда headline comparison становится слабым.

Но если переформулировать вопрос так:

> **Когда parametric adaptation вообще добавляет ценность сверх сильного компактного RAG на consumer hardware, и какой источник адаптера полезнее: supervised RAFT-style QLoRA или Doc-to-LoRA packaging?**

— то работа сразу становится гораздо сильнее. Тогда:

* **`S1`** — strong nonparametric baseline,
* **`S2+R`** — supervised retrieval-augmented adapter,
* **`S3/S4`** — попытка получить adapter без downstream QA supervision,
* **`S5`** — best practical hybrid / top-line.

В этой рамке pure parametric baselines (`S2`, `S3`, `S4` без retrieval) не исчезают, но становятся **контрольными опытами на границе применимости**, а не центральной осью. И это нормально. Более того, это часто сильнее: если такие baselines проваливаются, ты не “ломаешь thesis”, а показываешь, **что retrieval здесь — не роскошь, а необходимый компонент памяти**. Это ещё и полностью соответствует твоему evaluation spec: там прямо зафиксировано, что если `S1` beat’ит все parametric systems, это всё равно валидный результат, а не провал работы. 

## 2) Что здесь сейчас выглядит реально слабым

Самое слабое место в текущем дизайне — не данные и уже не размер корпуса, а **storyline**.

### Слабое место A: Axis 1 слишком претенциозен

Сейчас `S2 closed-book`, `S3 mono`, `S4-doc`, `S4-cluster` оформлены как равноправные системы для “paradigm comparison”, но по факту:

* `S2` у тебя уже **заранее признан ограниченным** 150 QA over 8 docs,
* `S3` зависит от качества merge 8 адаптеров в один,
* `S4` в большой степени отвечает не на вопрос про “memory paradigm”, а на вопрос “насколько routing спасает D2L merge bottleneck”.

Это интересные вещи, но они не все равны по научному весу. `S2` — контроль на пределы supervised memorization, `S3/S4` — исследование D2L packaging/routing, а не симметричный соперник RAG. Текущий `SPEC-systems` это почти уже говорит между строк. 

### Слабое место B: S5 сейчас логически awkward

По `SPEC-systems` и `STATE`, `S5` выбирает лучший **single adapter из S2 или S3** на том же `eval-50`, на котором потом и оценивается. Ты это честно задокументировал как mild optimistic bias, но именно это тоже добавляет ощущение “жидкости”: система выглядит как поздняя надстройка над уже проведёнными опытами, а не как естественная часть main story. Плюс `S2+R` уже существует как retrieval-augmented supervised baseline, так что `S5` легко начинает дублировать его по смыслу.   

## 3) Что делать дальше

Мой совет: **не спасать Axis 1 как главный narrative.**
Он уже выполнил свою работу: дал тебе conceptual calibration. Теперь нужно выбрать **одну главную историю** и под неё подстроить системы.

Я бы предложил такой поворот.

### Новый главный вопрос

**На компактном legal benchmark и на consumer hardware, даёт ли parametric adapter дополнительную ценность сверх сильного RAG, и какой способ получения адаптера более полезен — supervised RAFT-style QLoRA или Doc-to-LoRA?**

Это уже не “банальное навешивание улучшений к RAG”. Это:

* сравнение **двух источников parametric bias** поверх одного retrieval backbone;
* плюс измерение, можно ли вообще частично обойтись без retrieval;
* плюс сравнение стоимости, grounding и latency.

Это исследование **не про абстрактные виды памяти в вакууме**, а про **роль parametric memory в реальном document-grounded QA**. И это намного честнее.

## 4) Как я бы перестроил систему прямо сейчас

Я бы разделил всё на **headline systems** и **controls**.

### Headline systems

Это то, что реально должно попадать в main results table.

1. **S1 Classical RAG**
   Твой сильный nonparametric baseline. Он уже готов.  

2. **S2+R QLoRA RAFT + retrieval**
   Это уже готовый retrieval-augmented supervised baseline, и он уже показывает meaningful gain. Это должен быть один из главных героев, а не боковая ветка.  

3. **S3+R Doc-to-LoRA + retrieval**
   Этой системы в текущем SSOT по сути нет как headline, и это, на мой взгляд, главный пробел. Если Doc-to-LoRA вообще должен жить в работе как practically relevant method, его надо сравнивать **не только в closed-book режиме**, а ещё и как **generator adapter inside RAG**. Иначе ты рискуешь доказать лишь то, что “D2L без retrieval хуже retrieval”, что почти неинтересно.
   То есть D2L стоит тестировать в двух ролях:

   * как pure parametric limit (`S3` / `S4`) — контроль;
   * как retrieval-conditioned generator adapter (`S3+R`) — реальный contender.

4. **Final best practical hybrid**
   Либо явный `RAG + QLoRA`, либо `RAG + D2L`, либо “best of the two”, но без selection games на том же eval-50. Лучше заранее объявить обе системы и сравнить их напрямую.

### Controls / secondary analysis

Это уже не headline contenders, а то, что делает работу научно глубже.

* **S2 closed-book QLoRA** — negative control for pure supervised parametric memory
* **S3 monolithic D2L** — control for pure hypernetwork parametric memory
* **S4-doc / S4-cluster** — inner study on routing vs merge for D2L

Вот в такой конфигурации closed-book QLoRA уже не выглядит “зачем это вообще”, потому что он становится **вопросом о пределе внутренней памяти**, а не главным претендентом на победу.

## 5) Значит ли это, что работа теперь “про навешивание улучшений к RAG”?

Частично — да. Но это **не плохо**.

Плохой вариант звучал бы так:

> “мы просто прикрутили ещё один адаптер к RAG и получили чуть лучше.”

Хороший вариант звучит так:

> “мы сравнили, какие формы parametric adaptation вообще оправданы поверх сильного retrieval baseline на компактном legal corpus under consumer hardware, и отдельно измерили, где retrieval остаётся незаменимым.”

Это уже совсем другой уровень формулировки.

На самом деле многие хорошие прикладные работы именно так и устроены:

* сильный nonparametric baseline,
* одна-две parametric augmentation стратегии,
* чёткое cost/quality/grounding сравнение,
* честные negative controls.

У тебя для этого уже есть основа: benchmark компактный, S1 и S2+R уже живые, D2L feasible на уровне одного документа, и evaluation протокол уже готов к trade-off discussion.  

## 6) Мой жёсткий практический совет

Я бы сейчас сделал так:

**Оставить как core:**

* `S1 Classical RAG`
* `S2+R QLoRA RAFT`
* `S3+R D2L + retrieval`  ← добавить как обязательный headline system
* optional final “best practical system”

**Понизить до controls:**

* `S2 closed-book`
* `S3 mono`
* `S4-doc`
* `S4-cluster`

**Изменить главный RQ:**
не “кто победит в изоляции”, а
**“какую добавочную ценность даёт parametric adaptation сверх сильного compact RAG, и какой источник адаптера лучше работает в retrieval-grounded QA?”**

Вот тогда вся архитектура сразу становится собранной.

## 7) Если совсем коротко

Мой ответ на твои два вопроса:

**1. Актуален ли исходный вопрос?**
В текущей буквальной формулировке — **уже не очень**.
В переосмысленной формулировке про **limits of pure parametric memory + value of retrieval augmentation** — **да, вполне**.

**2. Что делать дальше?**
Не пытаться спасать closed-book QLoRA как contender.
Сделать его negative control.
Поднять `S2+R` в main story.
Добавить `D2L + retrieval` как симметричный headline contender.
И переписать SSOT так, чтобы **Axis 2 стал главным**, а **Axis 1 — калибровочной / falsification веткой**.

В таком виде работа, на мой взгляд, **не разваливается**, а наоборот становится заметно сильнее и честнее.