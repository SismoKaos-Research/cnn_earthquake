# EK- BODT İSTASYONU KAOS ÖZNİTELİKLERİ ANALİZ RAPORU

Bu rapor `bodt_q1_chaos_5hz_features.parquet` dosyası kullanılarak hazırlanmıştır. BODT istasyonuna ait sürekli sismik kayıtlar 1 Mayıs 2024 – 28 Ekim 2024 dönemini kapsamakta olup, 100 Hz'lik ham veri 5 Hz'e indirgenmiş, 200 saniyelik pencerelere bölünmüş ve her pencere kendi içinde normalize edildikten sonra kaos ve entropi öznitelikleri hesaplanmıştır. Elde edilen 312.626 pencere × 134 sütunluk matris, üç bileşen (Z, N, E) üzerinden Wolf ve Rosenstein Lyapunov üsleri, korelasyon boyutu, permütasyon ve örnek entropisi, Hjorth parametreleri ile spektral ve istatistiksel ölçütleri içermektedir.

## 1. Saatlik toplama

Öznitelik çıkarımı saatte 72 pencere üretmektedir. Bağlayıcı kısıt zamansal çözünürlük değil bağlam uzunluğu olduğundan — 24 adımlık bir dizi doğal çözünürlükte yalnızca 20 dakikayı kapsamaktadır — pencereler saatlik ortalama, standart sapma, en küçük ve en büyük değerlere indirgenmiştir. Sonuçta **4.343 saatlik satır ve 528 öznitelik sütunu** elde edilmiştir.

## 2. Etiket ve karşılaştırma tabanı

Etiket, istasyondan belirli bir uzaklık içinde ve belirli bir büyüklüğün üzerinde bir depremin **ilerideki** zaman aralığında gerçekleşip gerçekleşmediğidir. Aralık `(t, t + ufuk]` biçiminde yarı açıktır: tam olarak `t` anındaki bir olay girdi penceresinde zaten gözlenebilir olduğundan, bunu gelecek saymak etiketin kendi girdisini okuması anlamına gelirdi.

Bu çalışmanın yöntemsel çekirdeği, her sonucun bir **karşılaştırma tabanına** karşı raporlanmasıdır. Kullanılan taban *kalıcılık* (persistence) kuralıdır: son uygun depremden bu yana geçen süre. Bu kural hiçbir dalga formu bilgisi kullanmaz, öğrenme içermez ve sismik aktivitenin kümelenme eğiliminden başka bir şeye dayanmaz. Bir modelin taban değerinin üzerine ne kadar çıktığı, **kazanılan pay** olarak `(AUC − taban) / (1 − taban)` biçiminde ölçülmüştür.

Değerlendirme 4 katlı ileriye yürüyen (walk-forward) bölümleme ile, katlar arasında 24 saatlik ambargo bırakılarak yapılmıştır.

## 3. Sonuçlar

### 3.1 Etiket şemasının etkisi

Düz eşik (400 km içinde M≥2.5) fiziksel olarak tutarsızdır: 20 km'deki bir M2.5 istasyonun gürültü tabanının çok üzerindeyken 380 km'deki bir M2.5 bu tabanın altındadır, ancak düz etiket ikisini de pozitif saymaktadır. Uzaklığa göre kademelendirilmiş eşik bu tutarsızlığı gidermektedir. Çizelge 1, yalnızca kaos öznitelikleriyle eğitilen LightGBM modelinin dört şema altındaki sonuçlarını vermektedir.

| Etiket şeması | Pozitif | Taban | Model | Kazanılan pay | Kat bazında pay |
|---|---|---|---|---|---|
| Düz M≥2.5 / 400 km | %39,9 | 0,5714 | 0,5605 | **−%3,5** | +9,8 −20,5 −5,0 +1,6 |
| Kademeli, <100 km tüm M | %56,4 | 0,5318 | 0,5376 | +%1,2 | +1,4 −1,5 +2,2 +2,8 |
| Kademeli, iç eşik M≥2,0 | %23,4 | 0,5232 | 0,5542 | +%6,2 | +10,6 +2,9 −2,6 +14,0 |
| Kademeli, iç eşik M≥2,5 | %10,5 | 0,5752 | 0,6068 | **+%7,0** | +25,2 −6,0 +9,6 −0,9 |

Çizelge 1. Etiket şemasına göre kaos öznitelikleri (6 saatlik ufuk, 4 kat).

Ortalama değer taban altından taban üstüne çıkmakla birlikte, **hiçbir şema dört katın tamamında pozitif değildir**. +%7,0'lik en iyi sonuç, tek bir katın +%25,2 değerinden kaynaklanmakta, diğer iki kat sıfırın altında kalmaktadır. Düz şemanın katları ise 35 puanlık bir aralığa yayılmaktadır. Dört kat ve tek istasyon ile bu farkların hiçbiri gürültüden ayırt edilebilir değildir.

Kademeli şemanın 100 km içindeki tüm büyüklükleri kapsayan biçimi, saatlik adım ve 6 saatlik ufuk altında saatlerin %56,4'ünü pozitif yapmakta, etiket doygunlaşmakta ve taban 0,53'e düşmektedir. Bu şema 200 saniyelik pencereler için tasarlanmıştır; saatlik hücrede anlamlı kalabilmesi için bir iç büyüklük eşiği gerekmektedir.

### 3.2 Yinelemeli mimariler

Aynı soru üç dizi modeliyle, M≥4,0 ve 14 günlük hücrede yeniden ölçülmüştür (Çizelge 2).

| Model | Ortalama AUC | Kat SS | Taban | Kazanılan pay |
|---|---|---|---|---|
| LSTM | 0,5244 | 0,1051 | 0,5823 | −%13,87 |
| GRU | 0,5709 | 0,1621 | 0,5823 | −%2,73 |
| TCN | 0,5204 | 0,0698 | 0,5823 | −%14,82 |

Çizelge 2. Dizi modellerinin kalıcılık tabanına karşı başarımı.

Üç mimari de ortalamada kalıcılık kuralına yenilmektedir ve katlar arası standart sapma (0,07–0,16) her farkı aşmaktadır. Tek değişkenli tarama da aynı yöne işaret etmektedir: 528 özniteliğin yalnızca 78'i (%14,8) tabanı geçmektedir.

### 3.3 Görev tanımının etkisi

Aynı öznitelikler, aynı mimari ve aynı satırlarla yalnızca hedef değişkenin tanımı değiştirildiğinde ortaya çıkan fark Çizelge 3'te verilmiştir.

| Hedef değişken | AUC | Taban |
|---|---|---|
| Eşzamanlı — bu saat içinde olay var mı | **0,8347 ± 0,0095** | 0,5484 |
| İleriye dönük — 6 saat içinde olay olacak mı | 0,5652 ± 0,0194 | 0,5500 |

Çizelge 3. Eşzamanlı sınıflandırma ile ileriye dönük tahminin karşılaştırması.

Aradaki 0,27'lik AUC farkı yalnızca etiketin zamanından kaynaklanmaktadır. Kaos ve entropi öznitelikleri, **içinde deprem bulunan bir pencereyi arka plan gürültüsünden ayırmakta güçlüdür; bir depremi önceden haber vermemektedir.**

## 4. Değerlendirme

Bulgular üç maddede özetlenebilir. Birincisi, bir karşılaştırma tabanı olmadan hiçbir AUC değeri yorumlanamaz; kalıcılık kuralı bu veri üzerinde 0,52–0,58 aralığında AUC üretmektedir ve bir modelin bu bandın içinde kalması başarı değildir. İkincisi, etiket tasarımının iyileştirilmesi — uzaklığa göre kademeli büyüklük eşiği daha tutarlı bir tanımdır — sonucu değiştirmemekte, yalnızca daha sağlam kılmaktadır. Üçüncüsü ve en önemlisi, eşzamanlı sınıflandırma ile ileriye dönük tahmin aynı ölçüt altında rapor edilse bile aynı görev değildir; ikisi arasındaki fark bu veri üzerinde 0,27 AUC'dir.

Sınırlılıklar açıkça belirtilmelidir: sonuçlar tek istasyona (BODT), tek bir altı aylık döneme ve tek tohuma dayanmaktadır. Katlar arası değişkenlik gözlenen farkların çoğundan büyüktür. Bu nedenle rapor, kaos özniteliklerinin tahmin gücü olmadığını kanıtlamamakta; bu veri hacminde ve bu hücrede kalıcılık tabanını aşan bir sinyal **ölçülemediğini** belirtmektedir.
