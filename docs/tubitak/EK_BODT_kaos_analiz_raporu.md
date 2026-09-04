# EK- BODT İSTASYONU KAOS ÖZNİTELİKLERİ ANALİZ RAPORU

Bu rapor `bodt_q1_chaos_5hz_features.parquet` dosyası kullanılarak hazırlanmıştır. BODT istasyonuna ait sürekli sismik kayıtlar 1 Mayıs 2024 – 28 Ekim 2024 dönemini kapsamakta olup, 100 Hz'lik ham veri 5 Hz'e indirgenmiş, 200 saniyelik pencerelere bölünmüş ve her pencere kendi içinde normalize edildikten sonra kaos ve entropi öznitelikleri hesaplanmıştır. Elde edilen 312.626 pencere × 134 sütunluk matris, üç bileşen (Z, N, E) üzerinden Wolf ve Rosenstein Lyapunov üsleri, korelasyon boyutu, permütasyon ve örnek entropisi, Hjorth parametreleri ile spektral ve istatistiksel ölçütleri içermektedir.

## 1. Saatlik toplama

Öznitelik çıkarımı saatte 72 pencere üretmektedir. Bağlayıcı kısıt zamansal çözünürlük değil bağlam uzunluğu olduğundan — 24 adımlık bir dizi doğal çözünürlükte yalnızca 20 dakikayı kapsamaktadır — pencereler saatlik ortalama, standart sapma, en küçük ve en büyük değerlere indirgenmiştir. Sonuçta **4.343 saatlik satır ve 528 öznitelik sütunu** elde edilmiştir.

## 2. Etiket şemaları

Etiket, istasyondan belirli bir uzaklık içinde ve belirli bir büyüklüğün üzerinde bir depremin gerçekleşip gerçekleşmediğidir. Bir etiket tanımı iki ayrı seçimden oluşmaktadır: **hangi depremlerin sayılacağı** (uzaklık ve büyüklük) ve **hangi zaman aralığının sorulduğu** (eşzamanlı ya da ileriye dönük). Bu iki seçim birbirinden bağımsızdır ve raporun ilerleyen bölümlerinde ayrı ayrı incelenmektedir.

### 2.1 Zaman aralığı

**İleriye dönük etiket.** Sorulan soru "önümüzdeki *H* saat içinde uygun bir deprem olacak mı" biçimindedir. Aralık `(t, t + H]` olarak yarı açıktır: tam olarak `t` anındaki bir olay girdi penceresinde zaten gözlenebilir olduğundan, bunu gelecek saymak etiketin kendi girdisini okuması anlamına gelirdi. Bu rapordaki temel görev budur ve *H* = 6 saattir.

**Eşzamanlı etiket.** Sorulan soru "bu saat içinde uygun bir deprem var mı" biçimindedir. Bu bir tahmin değil, bir tespit görevidir. Yalnızca Çizelge 5'teki karşılaştırma için kullanılmıştır.

### 2.2 Uygun depremlerin seçimi

**Düz eşik (bu çalışmanın yayımlanmış tanımı).** İstasyondan 400 km yarıçap içinde M≥2,5 olan tüm depremler sayılır. Bu eşik 2026-08-21 tarihinde istatistiksel güce göre seçilmiştir: 232 olay, %25 pozitif oran, 0,543 kalıcılık tabanı. Daha dar yarıçaplar fiziksel olarak tercih edilebilir olmakla birlikte, ilgili dönemde anlamlı bir ölçüm yapılamayacak kadar az olay içermektedir.

Bu tanımın bilinen zayıflığı fiziksel tutarsızlığıdır: 20 km uzaklıktaki bir M2,5 istasyonun gürültü tabanının çok üzerinde bir iz bırakırken, 380 km uzaklıktaki bir M2,5 bu tabanın altında kalmaktadır. Düz eşik ikisini de aynı biçimde pozitif saymaktadır.

**Kademeli eşik.** Büyüklük alt sınırı uzaklıkla birlikte yükseltilerek yaklaşık olarak sabit bir tespit edilebilirlik hedeflenmektedir. Sırdar vd. (AIMSA 2026) tarafından ELBA istasyonu için önerilen şema Çizelge 1'de verilmiştir.

| Uzaklık aralığı | Alt büyüklük sınırı |
|---|---|
| 0 – 100 km | tüm büyüklükler |
| 100 – 300 km | M ≥ 3,0 |
| 300 – 500 km | M ≥ 5,0 |
| 500 – 1000 km | M ≥ 6,0 |

Çizelge 1. Uzaklığa göre kademeli büyüklük eşiği.

Şema bir üst sınır içermektedir: en dış yarıçapın ötesindeki bir deprem, büyüklüğü ne olursa olsun dışarıda bırakılmaktadır. Bu, uzak bir büyük depremin, istasyonda kendi gürültüsünün ancak biraz üzerinde bir iz bırakmasına rağmen pozitif sayılmasını önlemektedir.

**İç eşikli kademeli varyantlar.** Yukarıdaki şemanın 100 km içinde tüm büyüklükleri kapsayan iç bandı, 200 saniyelik pencereler için tasarlanmıştır. Saatlik adım ve 6 saatlik ufuk altında bu band saatlerin %56,4'ünü pozitif yapmakta, etiket doygunlaşmakta ve kalıcılık tabanı 0,53'e düşmektedir. Bu nedenle iç banda bir büyüklük alt sınırı eklenen iki varyant daha ölçülmüştür: iç eşik M≥2,0 (dış bandlar 3,0 / 5,0 / 6,0) ve iç eşik M≥2,5 (dış bandlar 3,5 / 5,0 / 6,0).

## 3. Karşılaştırma tabanı

Bu çalışmanın yöntemsel çekirdeği, her sonucun bir **karşılaştırma tabanına** karşı raporlanmasıdır. Kullanılan taban *kalıcılık* (persistence) kuralıdır: son uygun depremden bu yana geçen süre. Bu kural hiçbir dalga formu bilgisi kullanmaz, öğrenme içermez ve sismik aktivitenin kümelenme eğiliminden başka bir şeye dayanmaz. Bir modelin taban değerinin üzerine ne kadar çıktığı **kazanılan pay** olarak `(AUC − taban) / (1 − taban)` biçiminde ölçülmüştür.

Taban, etiketle birlikte değişmektedir. Etiket şeması değiştirildiğinde hem pozitif sınıf hem de taban aynı anda kaydığından, iki şemanın ham AUC değerleri doğrudan karşılaştırılamaz; karşılaştırılabilir olan kazanılan paydır.

Değerlendirme 4 katlı ileriye yürüyen (walk-forward) bölümleme ile, katlar arasında 24 saatlik ambargo bırakılarak yapılmıştır.

## 4. Model mimarileri

**LightGBM (temel model).** İkili hedef, AUC ölçütü, öğrenme oranı 0,03, yaprak sayısı 7, yaprak başına en az 80 örnek, öznitelik örnekleme oranı 0,5, torbalama oranı 0,7 (her turda), L2 düzenlileştirme 10,0 ve 300 tur. Yaprak sayısının küçük ve düzenlileştirmenin güçlü tutulması bilinçlidir: 528 sütun ve 4.343 satırla, daha esnek bir ağaç yapısı kat içi gürültüyü ezberlemektedir.

**Lojistik regresyon (doğrusal kontrol).** L2 cezası, C = 0,05, en fazla 2000 yineleme. Ağaç modelinin elde ettiği payın doğrusal olmayan bir yapıdan mı yoksa basit bir eşikten mi geldiğini ayırmak için birlikte raporlanmaktadır.

**LSTM.** Çift yönlü LSTM, yön başına 64 gizli birim (çıktı genişliği 128), ardından 4 başlıklı çok başlı öz-dikkat katmanı ve LayerNorm. Dikkat katmanı, dizinin hangi adımlarının ağırlıklandırılacağını öğrenmekte, böylece sabit uzunlukta bir gömme elde edilmektedir. Bırakma oranı 0,3.

**GRU.** Tek katmanlı GRU, 64 gizli birim, ardından tek çıkışlı doğrusal bir dikkat katmanı ile adım ağırlıklandırma. Başlık 64 → 32 → 1 biçiminde iki doğrusal katmandan oluşmakta ve aralarında 0,3 bırakma uygulanmaktadır. PyTorch'ta GRU'nun kendi bırakma parametresi yalnızca çok katmanlı yığınlarda katmanlar arasında etkili olduğundan, tek katmanlı yapıda bırakma başlıkta uygulanmıştır.

**TCN.** Üç seviyeli genişletilmiş (dilated) evrişim yığını, seviye başına 64 kanal, çekirdek boyu 3, bırakma 0,3. Her seviye iki Conv1d katmanı içermekte ve genişletme katsayısı seviyeyle birlikte katlanarak artmaktadır; böylece yinelemeli bağlantı kullanmadan uzun bağlam elde edilmektedir.

Çizelge 2, kullanılan modelleri özetlemektedir.

| Model | Yapı | Genişlik | Bırakma | Not |
|---|---|---|---|---|
| LightGBM | 300 turluk ağaç topluluğu | 7 yaprak | — | L2 = 10,0; öznitelik oranı 0,5 |
| Lojistik regresyon | doğrusal | — | — | L2, C = 0,05 |
| LSTM | çift yönlü + öz-dikkat | 64 (×2 yön) | 0,3 | 4 dikkat başlığı, LayerNorm |
| GRU | tek katman + dikkat | 64 | 0,3 | başlık 64 → 32 → 1 |
| TCN | 3 seviye genişletilmiş evrişim | 64 kanal | 0,3 | çekirdek 3, katlanan genişletme |

Çizelge 2. Karşılaştırılan model mimarileri.

**Çizelge 5'teki karşılaştırmanın mimarisi.** Görev tanımının etkisini ölçen deneyde, meslektaşlarımızın raporlarındaki yapılandırma bilinçli olarak birebir kullanılmıştır: iki katmanlı LSTM, katman başına 64 gizli birim, bırakma 0,4, Adam optimizasyonu, öğrenme oranı 0,001, L2 1e-5, yığın boyu 64, pozitif sınıfa 5 kat ağırlık, 5 turluk sabırla erken durdurma ve en fazla 40 tur; bağlam uzunluğu 24 saat. Mimari ve hiperparametreler sabit tutulduğundan, gözlenen fark yalnızca etiketin zaman tanımına atfedilebilir.

## 5. Sonuçlar

### 5.1 Etiket şemasının etkisi

Çizelge 3, yalnızca kaos öznitelikleriyle eğitilen LightGBM modelinin dört şema altındaki sonuçlarını vermektedir.

| Etiket şeması | Pozitif | Taban | Model | Kazanılan pay | Kat bazında pay |
|---|---|---|---|---|---|
| Düz M≥2,5 / 400 km | %39,9 | 0,5714 | 0,5605 | **−%3,5** | +9,8 −20,5 −5,0 +1,6 |
| Kademeli, <100 km tüm M | %56,4 | 0,5318 | 0,5376 | +%1,2 | +1,4 −1,5 +2,2 +2,8 |
| Kademeli, iç eşik M≥2,0 | %23,4 | 0,5232 | 0,5542 | +%6,2 | +10,6 +2,9 −2,6 +14,0 |
| Kademeli, iç eşik M≥2,5 | %10,5 | 0,5752 | 0,6068 | **+%7,0** | +25,2 −6,0 +9,6 −0,9 |

Çizelge 3. Etiket şemasına göre kaos öznitelikleri (6 saatlik ufuk, 4 kat).

Ortalama değer taban altından taban üstüne çıkmakla birlikte, **hiçbir şema dört katın tamamında pozitif değildir**. +%7,0'lik en iyi sonuç, tek bir katın +%25,2 değerinden kaynaklanmakta, diğer iki kat sıfırın altında kalmaktadır. Düz şemanın katları ise 35 puanlık bir aralığa yayılmaktadır. Dört kat ve tek istasyon ile bu farkların hiçbiri gürültüden ayırt edilebilir değildir.

Kademeli şemanın doygunlaşan iç bandı (%2.2) burada da görülmektedir: taban 0,53'e düşerek kalıcılık kuralının ayırt ediciliğini yitirdiği noktaya yaklaşmaktadır.

### 5.2 Yinelemeli mimariler

Aynı soru üç dizi modeliyle, M≥4,0 ve 14 günlük hücrede yeniden ölçülmüştür (Çizelge 4).

| Model | Ortalama AUC | Kat SS | Taban | Kazanılan pay |
|---|---|---|---|---|
| LSTM | 0,5244 | 0,1051 | 0,5823 | −%13,87 |
| GRU | 0,5709 | 0,1621 | 0,5823 | −%2,73 |
| TCN | 0,5204 | 0,0698 | 0,5823 | −%14,82 |

Çizelge 4. Dizi modellerinin kalıcılık tabanına karşı başarımı.

Üç mimari de ortalamada kalıcılık kuralına yenilmektedir ve katlar arası standart sapma (0,07–0,16) her farkı aşmaktadır. Tek değişkenli tarama da aynı yöne işaret etmektedir: 528 özniteliğin yalnızca 78'i (%14,8) tabanı geçmektedir.

### 5.3 Görev tanımının etkisi

Aynı öznitelikler, aynı mimari ve aynı satırlarla yalnızca hedef değişkenin tanımı değiştirildiğinde ortaya çıkan fark Çizelge 5'te verilmiştir.

| Hedef değişken | AUC | Taban |
|---|---|---|
| Eşzamanlı — bu saat içinde olay var mı | **0,8347 ± 0,0095** | 0,5484 |
| İleriye dönük — 6 saat içinde olay olacak mı | 0,5652 ± 0,0194 | 0,5500 |

Çizelge 5. Eşzamanlı sınıflandırma ile ileriye dönük tahminin karşılaştırması.

Aradaki 0,27'lik AUC farkı yalnızca etiketin zamanından kaynaklanmaktadır. Kaos ve entropi öznitelikleri, **içinde deprem bulunan bir pencereyi arka plan gürültüsünden ayırmakta güçlüdür; bir depremi önceden haber vermemektedir.**

## 6. Değerlendirme

Bulgular üç maddede özetlenebilir. Birincisi, bir karşılaştırma tabanı olmadan hiçbir AUC değeri yorumlanamaz; kalıcılık kuralı bu veri üzerinde 0,52–0,58 aralığında AUC üretmektedir ve bir modelin bu bandın içinde kalması başarı değildir. İkincisi, etiket tasarımının iyileştirilmesi — uzaklığa göre kademeli büyüklük eşiği daha tutarlı bir tanımdır — sonucu değiştirmemekte, yalnızca daha sağlam kılmaktadır. Üçüncüsü ve en önemlisi, eşzamanlı sınıflandırma ile ileriye dönük tahmin aynı ölçüt altında rapor edilse bile aynı görev değildir; ikisi arasındaki fark bu veri üzerinde 0,27 AUC'dir.

Sınırlılıklar açıkça belirtilmelidir: sonuçlar tek istasyona (BODT), tek bir altı aylık döneme ve tek tohuma dayanmaktadır. Katlar arası değişkenlik gözlenen farkların çoğundan büyüktür. Bu nedenle rapor, kaos özniteliklerinin tahmin gücü olmadığını kanıtlamamakta; bu veri hacminde ve bu hücrede kalıcılık tabanını aşan bir sinyal **ölçülemediğini** belirtmektedir.
