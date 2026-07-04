# Worked examples — auto-picked candidates per fate

Hand-pick one per fate for the paper box; trim passage text as needed. Sorted by
question length (shortest first) so the cleanest specimens lead.

## culprit  (131 candidates)

**2wiki/natural/qwen · 844fe30c0bda11eba7f7acde48001122**
- Q: Where did Giacomo Feo's wife die?
- gold: Florence   ·   model answered: Beaulieu-sur-Loire
- sufficient family: {Beaulieu-sur-Loire}
  - [Beaulieu-sur-Loire] Beaulieu- sur- Loire is a commune in the Loiret department in north -central France. It is also the place of death of Jacques MacDonald, a French general who served in the Napoleonic Wars

**2wiki/natural/mistral · e6688a7e0baf11ebab90acde48001122**
- Q: Who is Philip Of Sicily's father?
- gold: Charles I of Naples   ·   model answered: Charles I of Sicily
- sufficient family: {Philip of Sicily}
  - [Philip of Sicily] Philip (born 1255/56, died 1277), of the Capetian House of Anjou, was the second son of King Charles I of Sicily and Countess Beatrice of Provence. He was at various times set up to become King of Sardinia, Prince of Ach...

**musique/natural/qwen · 2hop__197470_271394**
- Q: Where was Tyler MacDuff's child educated?
- gold: Blair High School   ·   model answered: Blair International Baccalaureate School
- sufficient family: {Dana MacDuff}
  - [Dana MacDuff] He was born in Pasadena, California, to the actor Tyler MacDuff and the former Beverlie May Anderson (born 1930), who divorced in 1961. He is named for his father's close friend, the actor Dana Andrews. He graduated in 1...

## disjoint coalition  (143 candidates)

**musique/natural/qwen · 2hop__728887_159109**
- Q: Where is Assomada located?
- gold: central Atlantic Ocean   ·   model answered: Not mentioned
- sufficient family: {Royal Oaks Country Club} | {RTC (Cape Verde)} | {Larraul} | {Republic of the Congo}
  - [Royal Oaks Country Club] Royal Oaks Country Club is a country club and subdivision in Houston, Texas, United States in the Alief community. The country club is located at 2910 Royal Oaks Club Drive, west of The Galleria. Sunrise Colony Company d...
  - [RTC (Cape Verde)] The station also broadcasts news, sports, television shows and recently broadcasts football or soccer coverages from Portugal and also from Brazil as well as Latin America but rarely around the world. The radio channel i...
  - [Larraul] Larraul is a town and municipality located in the province of Gipuzkoa, in the autonomous community of Basque Country, northern Spain.
  - [Republic of the Congo] The capital, Brazzaville, is located on the Congo River, in the south of the country, immediately across from Kinshasa, the capital of the Democratic Republic of the Congo.

**musique/natural/qwen · 2hop__256553_617289**
- Q: Who is the uncle of Liu Bin?
- gold: Liu Yin   ·   model answered: Liu Yan
- sufficient family: {Liu Yin (Southern Han)} | {Liu Rushi, Consort Dowager Zhao} | {Hans Jacob Hess, Banu Amela, Consort Dowager Zhao}
  - [Liu Rushi] Liu Rushi (; 1618–1664), also known as Liu Shi, Liu Yin and Yang Yin, was a Chinese courtesan and poet in the late Ming dynasty who married Qian Qianyi at the age of 25. She committed suicide on the death of her husband....
  - [Hans Jacob Hess] John Jacob Hess was born in Wald, Zurich, Switzerland on May 17, 1584 to Hans Heinrich Hess, a bailiff, (1534–1587) and Adelheid Kuntz (1546–1585). He had eight older full siblings: Margaretha, Christian, Matheus, Hans, ...
  - [Banu Amela] They trace their genealogy back to Amela bin Saba'a bin Yashjeb bin Ya'arib bin Qahtan who left Yemen after the fourth destruction of the Marib Dam around 200 B.C. They dwelled in Jordan and Syria settling the southern h...
  - [Consort Dowager Zhao] Very little is known about the future Consort Dowager Zhao's background. It is known that she was considered very beautiful and was favored by Liu Yan. In 920, she gave birth to his third son Liu Hongdu — who would effec...
  - [Liu Yin (Southern Han)] Liu Yin (劉隱) (874 – April 4, 911), formally Prince Xiang of Nanhai (南海襄王), later further posthumously honored Emperor Xiang (襄皇帝) with the temple name of Liezong (烈宗) by his younger brother Liu Yan, was a warlord late in...

**musique/natural/qwen · 2hop__13731_801817**
- Q: Where was Mary's betrothed born?
- gold: Nazareth   ·   model answered: Joseph
- sufficient family: {Mary, mother of Jesus} | {Mary, mother of Jesus}
  - [Mary, mother of Jesus] Mary resided in "her own house"[Lk.1:56] in Nazareth in Galilee, possibly with her parents, and during her betrothal — the first stage of a Jewish marriage — the angel Gabriel announced to her that she was to be the moth...
  - [Mary, mother of Jesus] According to the apocryphal Gospel of James, Mary was the daughter of Saint Joachim and Saint Anne. Before Mary's conception, Anne had been barren and was far advanced in years. Mary was given to service as a consecrated...

## holistic  (46 candidates)

**2wiki/natural/mistral · ea1fc06c0bda11eba7f7acde48001122**
- Q: Where does Karin Stoltenberg's husband work at?
- gold: United Nations   ·   model answered: Oslo City Council (in 2015)
- sufficient family: (none within bound)

**musique/natural/mistral · 2hop__458131_174974**
- Q: Who is the spouse of the Rabbit Hole's producer?
- gold: Keith Urban   ·   model answered: Blossom Films (production company of Rabbit Hole) is Nicole Kidman's company, so her spouse
- sufficient family: (none within bound)

**2wiki/natural/mistral · 6638110e0bdb11eba7f7acde48001122**
- Q: Where was the father of Alexandre Gaydamak born?
- gold: Moscow, USSR   ·   model answered: May 1976 (Alexandre Gaydamak) was born in France. However, his father Arc
- sufficient family: (none within bound)

## parametric  (51 candidates)

**2wiki/natural/qwen · 0bedb7e80bdc11eba7f7acde48001122**
- Q: Why did John Middleton Murry's wife die?
- gold: tuberculosis   ·   model answered: extrapulmonary tuberculosis
- sufficient family: (none within bound)

**hotpotqa/natural/phi · 5a7bbb64554299042af8f7cc**
- Q: Who is older, Annie Morton or Terry Richardson?
- gold: Terry Richardson   ·   model answered: Annie Morton is older; she was born in 1970, while Terry Richardson was born in
- sufficient family: (none within bound)

**hotpotqa/natural/mistral · 5a7320565542991f9a20c61d**
- Q: who is younger Keith Bostic or Jerry Glanville ?
- gold: Keith Bostic   ·   model answered: Jerry Glanville
- sufficient family: (none within bound)
