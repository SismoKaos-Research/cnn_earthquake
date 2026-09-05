# SÜREKLİ VERİ ÜZERİNDE YANLIŞ ALARM ORANI ANALİZ RAPORU

Bu rapor, MANT istasyonuna ait 728 günlük kesintisiz sismik kayıt üzerinde hazırlanmıştır. Kayıt 30 Nisan 2024 – 9 Ağustos 2026 dönemini kapsamakta olup 36 arşiv parçasından oluşmaktadır. Projede geliştirilen kısa pencereli deprem/gürültü sınıflandırıcıları ile klasik bir eşik yöntemi, aynı pencerelerde ve aynı ön işlemeyle karşılaştırılmıştır. 6 saniyelik model için 10.487.211, 3,4 saniyelik modeller için 18.519.887 pencere puanlanmıştır.

## Özet

- **Sınama kümesinde belirlenen 0,5 eşiği sürekli veride kullanılamamaktadır.** Öngörü günde 257 yanlış alarmdı; ölçülen değer 12.599'dur. 6 saniyelik model sakin bir istasyon gününün %92,7'sini işaretlemektedir.
- Nedeni ölçülmüştür: model, eğitim sırasında hiç görmediği bir **sessizlik bölgesinde** dışdeğerleme yapmaktadır ve sürekli arka plan tam olarak orada bulunmaktadır.
- **Çalışma noktası ölçülen arka plandan türetilmelidir.** Günde 10 yanlış alarm bütçesinde 6 saniyelik model, kaydedilmiş depremlerin %74,1'ini bulmaktadır (Bölüm 6).
- **AUC ile çalışma noktası ters sıralama vermektedir.** Klasik STA/LTA yöntemi AUC'de birinci, işletme bütçelerinde sonuncudur.
- Tespit edilen olayların **%69,6'sı S dalgasından önce** bildirilmektedir; ancak bu bir uzaklık koşuluna bağlıdır ve 25 km içinde her iki model de çoğunlukla geç kalmaktadır.
- Açıklanamayan alarmlar **günlük döngü** göstermektedir (gündüz/gece 1,7–2,1 kat), yani önemli bir bölümü kataloglanmamış deprem değil kültürel gürültüdür.

## 1. Sorunun tanımı

Bu projedeki bütün tespit sonuçları **düzenlenmiş bir sınama kümesi** üzerinde ölçülmüştür: sınıflar dengelidir, pozitif pencereler varış zamanına sabitlenmiştir ve negatif pencereler bir gürültü havuzundan genliğe göre seçilmiştir. Böyle bir kümede ölçülen başarım, koşullu bir niceliği yanıtlamaktadır: *pencerede bir varış olduğu bilindiğinde model onu bulabiliyor mu?*

Sürekli işletim ise koşulsuz niceliği gerektirmektedir: *rastgele bir pencerede varış var mı?* İki durumdaki taban oranlar arasında dört ila altı büyüklük mertebesi fark bulunmaktadır. Bu rapor, aradaki farkı doğrudan ölçmektedir.

Sınama kümesinin kendi öngörüsü şudur: 0,5 eşiğinde 6 saniyelik topluluk modeli 7.906 gürültü penceresinin 141'ini yanlış sınıflandırmakta, yani yanlış pozitif oranı %1,78'dir. Bir gün 14.400 ayrık 6 saniyelik pencere içerdiğinden, bu oran **günde 257 yanlış alarm** öngörmektedir.

## 2. Yöntem

### 2.1 Pencereleme

Pencereler ayrıktır (adım = pencere boyu). Böylece alarm sayısı aynı zamanda bağımsız karar sayısıdır. Hiçbir pencere veri boşluğu üzerinden kurulmamaktadır: pencereler yalnızca üç bileşenin de kesintisiz kayda sahip olduğu aralıklarda üretilmektedir. MANT'ta istasyon 21 günlük bir parça içinde yüzlerce kez düşüp geri gelmekte olduğundan bu kısıt önemsiz değildir.

### 2.2 Ön işleme ve doğrulama

Ön işleme, eğitim hattının kendi işlem sırasıdır: iki kez eğilim giderme, %5 Hann sönümlemesi, 4. derece 1–45 Hz bant geçiren süzgeç ve istasyonun uzun dönem gürültü tabanına göre standartlaştırma. Tek fark, işlemlerin tek pencere yerine pencere yığınlarına uygulanmasıdır.

Bu eşdeğerlik varsayılmamış, **ölçülmüştür**. Yığın süzgeci, `clean_and_filter_1d` işlevinin birebir kopyasına karşı en fazla 5,4·10⁻¹³ bağıl fark vermektedir. Ayrıca gerçek veri kümesi tensörleri, taramanın kendi puanlama yolundan geçirildiğinde yayımlanmış değerleri yeniden üretmektedir: 6 saniyelik model için 0,9899 (yayımlanmış 0,9896), P-dalgası modeli için 0,8742 (yayımlanmış 0,8712).

### 2.3 Duyarlılık yalnızca kaydedilmiş depremlerde sorulmaktadır

İstasyondan 500 km içindeki 47.522 katalog depremi için sinyal-gürültü oranı (SGO) ölçülmüştür. Ortanca SGO **1,39**'dur ve olayların yalnızca **%27**'si SGO 3 eşiğini aşmaktadır. Yani katalogdaki tipik deprem, MANT'ın ham kaydında görünür bir iz bırakmamaktadır.

Bütün katalog olaylarına karşı ölçüldüğünde olay düzeyinde AUC 0,67–0,73 çıkmaktadır. **Bu değer modelin değil, katalogun erişiminin ölçüsüdür.** Bu nedenle duyarlılık, yalnızca SGO≥3 koşulunu sağlayan 13.056 olay üzerinde raporlanmaktadır.

## 3. Karşılaştırma tabanı

Bir tespit ediciye ait "günde 10 alarm" değeri tek başına yorumlanamaz; ancak yerini alacağı yöntemin aynı kayıt üzerindeki başarımına karşı anlam kazanmaktadır. Bu nedenle klasik **STA/LTA** yöntemi dördüncü bir kol olarak aynı pencerelerde çalıştırılmıştır (STA 0,5 s, LTA 10 s, düşey bileşen).

Karakteristik işlev, pencere başına değil **sürekli parça üzerinde** hesaplanmaktadır: 10 saniyelik bir LTA 6 saniyelik pencereye sığmamaktadır ve pencere başına hesaplamak klasik yöntemi haksız biçimde zayıflatırdı. Her pencere, içindeki en yüksek karakteristik işlev değerini almaktadır. Böylece iki yöntem aynı pencerelerde, aynı süzgeçten geçmiş veriyle ve yalnızca tespit edici bakımından farklı olarak karşılaştırılmaktadır.

**Eşikler ölçülen arka plan dağılımından türetilmektedir.** Sınama kümesinin 0,5 eşiği sürekli veride anlamsızdır (Bölüm 5.1); bunun yerine kabul edilebilir günlük alarm bütçesi belirlenmekte ve o bütçeyi veren eşik arka plan dağılımının ilgili yüzdeliğinden okunmaktadır.

## 4. Model mimarileri

Karşılaştırılan dört kol Çizelge 1'de özetlenmiştir. Üç sinir ağı kolu da aynı mimariyi kullanmaktadır: şeritli evrişimlerle 8 kat indirgeme, ardından çift yönlü LSTM ve çok başlı öz-dikkat, sonunda ortalama havuzlama ve tek çıkışlı ikili baş (`ConvSeqBranch`, gizli boyut 48, 4 dikkat başı). Kollar yalnızca eğitildikleri veri kümesiyle ayrılmaktadır. Her kol 42, 43 ve 44 tohumlarıyla eğitilmiş üç modelin olasılık ortalamasıdır.

| Kol | Pencere | Eğitim kümesi | Negatif seçimi | Sınama AUC'si |
|---|---|---|---|---|
| 6s | [P−2,0; P+4,0] s | `catalog_6s_matched_hard` | genlik eşleştirmeli | 0,9896 |
| ponly | [P−2,0; P+1,4] s | `ponly_3p4s_matched` | genlik eşleştirmeli | 0,8712 |
| pnat | [P−2,0; P+1,4] s | `ponly_3p4s_natural` | madencilik yok | — |
| stalta | 6,0 s | — | — | — |

Çizelge 1. Karşılaştırılan tespit edici kolları.

**6s ile P-dalgası kolları arasındaki fark yalnızca pencere boyu değildir.** 6 saniyelik pencerede S dalgası olayların %28,8'inde, 25 km içindeki olayların ise %99,3'ünde pencere içine düşmektedir. P-dalgası kollarında 1,4 saniyelik kuyruk, S dalgasının hız modeline göre dışarıda kalmasını sağlamaktadır. Bu nedenle P-dalgası kolu bir **faz tespiti** görevi olarak okunmalıdır; erken uyarı olarak okunması, pencerenin kendisinin dayattığı bir uzaklık koşulunu gizlice kabul etmek anlamına gelirdi.

## 5. Sonuçlar

### 5.1 Sınama eşiği sürekli veriye aktarılamamaktadır

| Kol | Arka plan ortancası | 0,5 eşiğinde günlük alarm |
|---|---|---|
| 6s | **0,8019** | **12.599** |
| ponly | 0,4306 | 79 |
| pnat | 0,3037 | 45 |
| stalta | 1,49 (KİD) | 13.489 |

Çizelge 2. Arka plan puan dağılımı ve sınama eşiğinin sürekli veride ürettiği alarm sayısı.

6 saniyelik model sürekli **gürültü** üzerinde ortanca 0,80 puan vermekte ve sakin bir istasyon gününün %92,7'sini işaretlemektedir. Öngörülen 257 alarma karşılık ölçülen değer 12.599'dur; aradaki fark 49 kattır.

Ayrıca bu modelin kullanılabilir eşik bandı son derece dardır: arka planın %50'lik dilimi 0,797, %99,9'luk dilimi 0,839'dur. Eşiğin 0,0024 kadar kaydırılması alarm oranını on kat değiştirmektedir. Mevsimsel bir kaymanın bu bandı aşması beklenebilir.

### 5.2 Nedeni: genlik boşluğu

Gerçek eğitim **gürültü** pencereleri, tek bir çarpan dışında hiçbir şey değiştirilmeden modele verildiğinde:

| Ölçek | Ortanca standart sapma (istasyon σ) | p>0,5 oranı |
|---|---|---|
| 1,000 | 0,708 | 0,019 |
| 0,300 | 0,212 | 0,183 |
| 0,100 | 0,071 | 0,862 |
| 0,010 | 0,007 | 1,000 |

Çizelge 3. 6 saniyelik modelin genliğe tepkisi (gerçek gürültü pencereleri ölçeklenmiştir).

`P(deprem | genlik)` ilişkisi **U biçimlidir**. Genlik madenciliği negatif sınıfın altına bir taban koymakta, fiziğin kendisi pozitif sınıfın altına bir taban koymaktadır; sonuçta ~0,1 σ değerinin altında model **hiçbir sınıfa ait eğitim verisi görmemiş** olmakta ve dışdeğerlemesi "deprem" yönünde çıkmaktadır. Sürekli MANT arka planı tam olarak 0,11 σ'da bulunmaktadır.

İki olası açıklama sınanmış ve elenmiştir. Birincisi, taban σ'nın yanlış hesaplanmış olması: eğitim gürültü dosyalarındaki rastgele pencereler de aynı aralığa düşmekte, GADA (0,049), GELI (0,088) ve ENEZ (0,085) gibi eğitim istasyonları MANT'ın 0,11 değeriyle aynı bölgede yer almaktadır. İkincisi, genlik madenciliğinin kendisi: `ponly` madencilikli, `pnat` madenciliksizdir ve **her ikisi de tekdüzedir**. Kusur, S dalgasını içerdiği için pozitifleri 581 σ'ya kadar uzanan 6 saniyelik yapıya özgüdür.

### 5.3 AUC ile çalışma noktası ters sıralama vermektedir

| Kol | Olay AUC'si (SGO≥3) | Duyarlılık, 100 alarm/gün | 10 alarm/gün | 1 alarm/gün |
|---|---|---|---|---|
| stalta | **0,9795** | **0,928** | 0,548 | 0,132 |
| ponly | 0,9622 | 0,903 | 0,627 | 0,183 |
| pnat | 0,9516 | 0,861 | 0,573 | 0,219 |
| 6s | 0,9403 | 0,864 | **0,741** | **0,316** |

Çizelge 4. Olay düzeyinde ayırt etme gücü ile çalışma noktalarındaki duyarlılık.

**1978 tarihli STA/LTA yöntemi bütün SGO eşiklerinde AUC bakımından birinci, bütün alarm bütçelerinde ise sonuncudur.** Günde 1 alarm bütçesinde 6 saniyelik model 2,4 kat daha fazla deprem bulmaktadır.

Nedeni yöntemseldir: AUC bütün ROC eğrisini bütünlemektedir, oysa bir çalışma noktası eğrinin tek bir uç köşesinde yaşamaktadır — burada yanlış pozitif oranı 7,4·10⁻⁴'tür. Bir tespit edici genel sıralamada daha iyi, uç kuyrukta daha kötü ayrışabilmektedir. **Bu nedenle işletmeye yönelik hiçbir karşılaştırma yalnızca AUC ile raporlanmamalıdır.**

Günde 10 alarm bütçesindeki karışıklık matrisi Çizelge 5'te verilmiştir. Olay düzeyinde doğru negatif hücresi tanımsızdır: "negatif deprem" diye bir nesne yoktur. Alarmlar 60 saniyelik pencerede kümelenmektedir, böylece tek bir gürültü patlaması on yerine bir yanlış bildirim saymaktadır.

| Kol | DP | YN | YP | Kesinlik | Duyarlılık | F1 |
|---|---|---|---|---|---|---|
| 6s | 9.675 | 3.381 | 5.326 | 0,645 | **0,741** | **0,690** |
| ponly | 8.181 | 4.875 | 5.624 | 0,593 | 0,627 | 0,609 |
| pnat | 7.481 | 5.575 | 5.829 | 0,562 | 0,573 | 0,568 |
| stalta | 7.152 | 5.904 | 5.952 | 0,546 | 0,548 | 0,547 |

Çizelge 5. Günde 10 alarm bütçesinde olay düzeyinde karışıklık matrisi (SGO≥3, 13.056 olay).

### 5.4 Açıklanamayan alarmlar günlük döngü göstermektedir

| Kol | Gündüz (06–20) | Gece | Oran |
|---|---|---|---|
| 6s | 368/saat | 213/saat | 1,73 |
| ponly | 388/saat | 186/saat | 2,08 |
| pnat | 387/saat | 187/saat | 2,07 |
| stalta | 378/saat | 199/saat | 1,90 |

Çizelge 6. Günde 10 alarm bütçesinde açıklanamayan alarmların yerel saate göre dağılımı.

Bu rapordaki bütün yanlış alarm sayıları biçimsel olarak **üst sınırdır**: katalogda karşılığı bulunmayan bir alarm ya gerçek bir yanlış pozitiftir ya da AFAD'ın kataloglamadığı bir depremdir ve buradaki yöntem ikisini ayıramamaktadır. Ancak günlük döngü bu sınırı önemli ölçüde daraltmaktadır: **depremlerin mesai saatlerinde tepe yapması beklenmez.** Dört kolun tamamında 12:00–15:00 arasında gözlenen tepe, alarmların kayda değer bir bölümünün gerçekten kültürel gürültü olduğunu göstermektedir. Bu bulgu ancak iki yıla yayılan tam günler içeren bir kayıtta görünür hale gelmektedir.

### 5.5 P dalgasının S dalgasından önce bildirilmesi

Bir tespit, penceresinin tamamı gözlenip puanlanmadan bildirilemez. Bu nedenle **alarm zamanı pencerenin sonudur**, başlangıcı değil: *t* anında başlayan 6 saniyelik bir pencere *t*+6'da bildirim üretmektedir. Başlangıcın kullanılması, modele henüz sahip olmadığı bilgiyi atfetmek ve bazı olayları P varışından önce tespit edilmiş göstermek anlamına gelirdi.

Ayrık pencerelerde P varışı pencere içinde rastgele bir konuma düşmektedir. Ölçülen ortanca bildirim gecikmesi 6 saniyelik kol için P+4,5 s, 3,4 saniyelik kol için P+2,9 s'dir.

S−P farkı yaklaşık olarak *uzaklık* / 8,4 saniye olduğundan, bir tespit edicinin S dalgasını geçebilmesi için **S−P farkının kendi bildirim gecikmesini aşması** gerekmektedir. Buradan çıkan başabaş uzaklıklar 6 saniyelik kol için yaklaşık 38 km, P-dalgası kolu için yaklaşık 24 km'dir.

| Uzaklık aralığı | Ortanca S−P | 6s | ponly | stalta |
|---|---|---|---|---|
| 0 – 25 km | 1,4 s | 0,19 | **0,39** | 0,16 |
| 25 – 50 km | 4,0 s | 0,41 | **0,66** | 0,52 |
| 50 – 100 km | 10,7 s | 0,69 | 0,76 | **0,97** |
| 100 – 200 km | 15,1 s | 0,86 | 0,85 | **0,97** |
| 200 – 500 km | 32,1 s | 0,94 | 0,93 | **0,98** |

Çizelge 7. S dalgasından önce bildirilen olayların oranı (günde 10 alarm bütçesi, SGO≥3).

Ölçülen başabaş uzaklıklar, çizelgedeki %50 geçişleriyle birebir örtüşmektedir: 6 saniyelik kol 25–50 km bandında, P-dalgası kolu ise bu bandın altında %50'yi aşmaktadır. **Kısa pencere yaklaşık 14 km'lik ek kullanılabilir menzil kazandırmaktadır.** 25 km içinde ise her iki model de çoğunlukla geç kalmaktadır; S dalgası, model daha karar veremeden istasyona ulaşmaktadır.

Toplam oranlar Çizelge 8'de verilmiştir. Burada dikkat çeken nokta, klasik yöntemin en yüksek "S'den önce" oranına sahip olmasına karşın en az olayı bulmasıdır; bu, tespitlerinin uzak olaylara kaymasından kaynaklanmaktadır.

| Kol | Tespit edilen | S'den önce | Ortanca bildirim |
|---|---|---|---|
| 6s | 9.672 (%74,1) | 6.729 (%69,6) | P+4,5 s |
| ponly | 8.182 (%62,7) | 6.216 (%76,0) | P+2,9 s |
| stalta | 7.152 (%54,8) | 6.744 (%94,3) | P+4,3 s |

Çizelge 8. Toplam tespit ve öncelik oranları (günde 10 alarm bütçesi, SGO≥3, 13.056 olay).

50 km'nin ötesinde klasik yöntemin öncelik tutarlılığı belirgin biçimde yüksektir (%97–98). Ortanca öncelik süreleri benzer olduğuna göre (50–100 km bandında −6,3 s ve −5,8 s), fark dağılımın yayılımındadır: STA/LTA yalnızca keskin başlangıca tepki vermekte, sinir ağları ise zayıf olaylarda zaman zaman dalga formunun daha geç bölümlerine kilitlenmektedir. Bu, tespit başarımını etkilememekte ancak zamanlamayı dağıtmaktadır. **Öncelik süresi tutarlılığı açısından klasik yöntem daha güvenilirdir.**

### 5.6 Daha sık pencereleme öncelik sorununu çözmemektedir

Bölüm 5.5'teki bildirim gecikmesinin ne kadarının **ızgara etkisinden**, ne kadarının modelin gerçek gecikmesinden kaynaklandığı ayrı bir deneyle ölçülmüştür. Katalog olaylarının çevresi 0,5 saniyelik adımla yeniden taranmıştır; olaylar kaydın %1'inden azını kapladığından bu tarama dakikalar sürmektedir.

Aynı eşikte sonuç belirgindir: 6 saniyelik kolun ortanca bildirimi P+4,5 s'den **P+1,9 s**'ye, S'den önce bildirim oranı %69,6'dan **%90,7**'ye çıkmaktadır. Gecikmenin baskın bileşeni ızgara etkisidir.

Ancak bu kazanç bedelsiz değildir. Örtüşen pencereler birim zamanda yaklaşık 12 kat daha fazla karar üretmekte, dolayısıyla aynı eşikte yanlış bildirim oranı günde 7,3'ten yaklaşık 137'ye çıkmaktadır. Bütçe sabit tutulduğunda eşik yükseltilmek zorundadır ve karşılaştırma Çizelge 9'daki gibi olmaktadır.

| Yapılandırma | Eşik | Tespit oranı | S'den önce | Ortanca bildirim |
|---|---|---|---|---|
| 6s, ayrık pencere | 0,8569 | %74,1 | %69,6 | P+4,5 s |
| 6s, sık pencere | 0,8999 | %38,9 | %88,2 | P+2,2 s |
| stalta, sık pencere | 13,41 | %18,8 | **%97,4** | **P+1,4 s** |

Çizelge 9. Sık pencerelemenin eşit yanlış bildirim bütçesinde (günde ~10) etkisi.

İki sonuç çıkmaktadır. Birincisi, sık pencereleme sabit bütçede **duyarlılığı düşürmektedir**: eşik 0,857'den 0,900'e çıkmakta, bu da modelin en yüksek puanı olan 0,903'e çok yaklaşmaktadır. Bölüm 5.1'de belirtilen dar band sorunu burada doğrudan bağlayıcı hale gelmektedir.

İkincisi ve daha önemlisi, **klasik yöntemin öncelik üstünlüğü sık pencerelemede de sürmektedir** (P+1,4 s'ye karşı P+2,2 s; %97,4'e karşı %88,2). Bunun nedeni yapısaldır: STA/LTA'nın karakteristik işlevi doğrudan dalga başlangıcına tepki vermekte ve bağlam gerektirmemektedir; sinir ağları ise eğitildikleri pencere geometrisi kadar varış sonrası sinyal beklemektedir. Öncelik süresini iyileştirmenin yolu tarama biçimini değiştirmek değil, **varış sonrası kısmı daha kısa olan bir pencereyle yeniden eğitmektir** — bu da tespit başarımından ödün vermek anlamına gelmektedir.

Bu bölümün sayıları 6 parçalık bir alt kümeye (2.445 olay) dayanmaktadır ve eşik ayarı yalnızca 0,39 günlük temiz olay öncesi gürültüyle yapılmıştır; kuyruk kestirimi gürültülüdür.

## 6. Çalışma noktası seçimi

Bu bölüm, raporun işletmeye yönelik özetidir. Soru şudur: **günde kaç yanlış alarma katlanılabiliyorsa, karşılığında ne alınmaktadır?**

| Bütçe (yanlış alarm/gün) | Eşik | Tespit oranı | S'den önce | Ortanca bildirim |
|---|---|---|---|---|
| 100 | 0,8350 | %86,4 | %76,3 | P+4,3 s |
| **10** | **0,8569** | **%74,1** | **%69,6** | **P+4,5 s** |
| 1 | 0,8950 | %31,6 | %76,8 | P+4,1 s |
| 0,1 | 0,9014 | %6,4 | %91,7 | P+3,4 s |

Çizelge 10. 6 saniyelik model için bütçe–başarım dengesi (SGO≥3, 13.056 olay).

Çizelgenin okunuşu şöyledir. Günde 100 yanlış alarm — yaklaşık saatte dört — kabul edilebiliyorsa, kaydedilmiş depremlerin %86'sı bulunmaktadır. Bütçe günde 10'a indirildiğinde duyarlılık %74'e gerilemektedir. Günde 1 alarmda ise duyarlılık %32'ye düşmektedir: **bütçenin son bir mertebelik daralması duyarlılığın yarısından fazlasına mal olmaktadır.**

Dört kolun aynı bütçelerdeki duyarlılıkları Çizelge 4'te verilmiştir. Seçim, bütçeye göre değişmektedir: günde 100 alarmda klasik yöntem, günde 10 ve 1 alarmda ise 6 saniyelik model önde gelmektedir.

**Bu oranlar tek istasyon içindir.** İşletimdeki erken uyarı sistemleri en az iki istasyonun uyuşmasını şart koşmaktadır. Bağımsız istasyonlarda yanlış alarmların çakışma olasılığı çarpım kuralıyla düştüğünden, ağ düzeyinde bu bütçeler çok daha rahat karşılanabilmektedir; buradaki değerler bir ağ için **üst sınır** niteliğindedir.

## 7. Değerlendirme

Bulgular dört maddede özetlenebilir.

Birincisi, **düzenlenmiş sınama kümesinde belirlenen eşik sürekli veriye aktarılamamaktadır.** 6 saniyelik modelin sınama kümesindeki 0,9896 AUC değeri geçerliliğini korumakla birlikte, aynı kümede belirlenen 0,5 eşiği sürekli kayıtta sakin bir günün %92,7'sini işaretlemektedir. Eşik, ölçülen arka plan dağılımından türetilmek zorundadır.

İkincisi, bunun nedeni ölçülmüştür: model, negatiflerin genliğe göre seçilmesi nedeniyle eğitim sırasında hiç görmediği bir sessizlik bölgesinde dışdeğerleme yapmaktadır ve sürekli arka plan tam olarak o bölgede bulunmaktadır. Kusur pencere tasarımına özgüdür, genlik madenciliğinin kendisine değil.

Üçüncüsü ve yöntemsel olarak en genel bulgu, **AUC ile çalışma noktasının ters sıralama verebilmesidir.** Bu veri üzerinde dört tespit ediciyi AUC ile sıralamak, işletme koşullarındaki sıralamanın tam tersini vermektedir. Bulgu bu modellere özgü bir tuhaflık değil, ölçütün tanımından çıkan bir sonuçtur.

Dördüncüsü, açıklanamayan alarmların günlük döngüsü, bunların önemli bir bölümünün kataloglanmamış deprem değil kültürel gürültü olduğunu göstermektedir.

Beşincisi, S dalgasından önce bildirim yapabilme yeteneği bir uzaklık koşuluna bağlıdır ve bu koşul pencere boyunun kendisinden doğmaktadır. 25 km içinde her iki model de çoğunlukla geç kalmaktadır. Bu nedenle P-dalgası kolu bir erken uyarı bileşeni olarak değil, uzaklık tabanı bulunan bir faz tespiti bileşeni olarak değerlendirilmelidir.

Altıncısı, öncelik süresi tarama sıklığıyla iyileştirilememektedir. Sık pencereleme mutlak gecikmeyi azaltmakta, ancak sabit bütçede duyarlılığı düşürmekte ve klasik yöntemin üstünlüğünü ortadan kaldırmamaktadır. Bu bir ayar sorunu değil, mimari bir sınırdır: karar için varış sonrası sinyal gerektiren bir model, doğrudan başlangıca tepki veren bir orana yetişememektedir.

**Sınırlılıklar açıkça belirtilmelidir.** Sonuçlar tek istasyona (MANT) dayanmaktadır; istasyonlar arası değişkenlik ölçülmemiştir ve alan yazında bu değişkenliğin makine öğrenmesi tabanlı seçicilerde arttığına dair bulgular bulunmaktadır. Ölçüm tek istasyonlu ve ilişkilendirme (association) öncesidir; işletimdeki bütün erken uyarı sistemleri yanlış alarmları birden çok istasyonun uyuşmasını şart koşarak bastırmaktadır, dolayısıyla buradaki oranlar bir ağ için üst sınırdır. Gecikme ayrıştırması yapılmamıştır; yalnızca model terimi bilinmektedir, çünkü bir tespit ancak pencerenin tamamı gözlendikten sonra bildirilebilmektedir. Son olarak sinir ağı çıkışları kalibre edilmiş olasılık değildir ve bu çalışmada kalibre edilmemiştir.

Rapor, kısa pencereli sınıflandırıcıların sürekli veride kullanılamayacağını göstermemektedir. Gösterdiği şey, **düzenlenmiş bir sınama kümesinde belirlenen çalışma noktasının sürekli veriye taşınamayacağı** ve tespit edicilerin yalnızca toplulaştırılmış bir ölçütle karşılaştırılmasının yanıltıcı olabileceğidir.
