# EK- SÜREKLİ VERİ ÜZERİNDE YANLIŞ ALARM ORANI ANALİZ RAPORU

Bu rapor, MANT istasyonuna ait 728 günlük kesintisiz sismik kayıt üzerinde hazırlanmıştır. Kayıt 30 Nisan 2024 – 9 Ağustos 2026 dönemini kapsamakta olup 36 arşiv parçasından oluşmaktadır. Projede geliştirilen kısa pencereli deprem/gürültü sınıflandırıcıları ile klasik bir eşik yöntemi, aynı pencerelerde ve aynı ön işlemeyle karşılaştırılmıştır. 6 saniyelik model için 10.487.211, 3,4 saniyelik modeller için 18.519.887 pencere puanlanmıştır.

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

## 6. Değerlendirme

Bulgular dört maddede özetlenebilir.

Birincisi, **düzenlenmiş sınama kümesinde belirlenen eşik sürekli veriye aktarılamamaktadır.** 6 saniyelik modelin sınama kümesindeki 0,9896 AUC değeri geçerliliğini korumakla birlikte, aynı kümede belirlenen 0,5 eşiği sürekli kayıtta sakin bir günün %92,7'sini işaretlemektedir. Eşik, ölçülen arka plan dağılımından türetilmek zorundadır.

İkincisi, bunun nedeni ölçülmüştür: model, negatiflerin genliğe göre seçilmesi nedeniyle eğitim sırasında hiç görmediği bir sessizlik bölgesinde dışdeğerleme yapmaktadır ve sürekli arka plan tam olarak o bölgede bulunmaktadır. Kusur pencere tasarımına özgüdür, genlik madenciliğinin kendisine değil.

Üçüncüsü ve yöntemsel olarak en genel bulgu, **AUC ile çalışma noktasının ters sıralama verebilmesidir.** Bu veri üzerinde dört tespit ediciyi AUC ile sıralamak, işletme koşullarındaki sıralamanın tam tersini vermektedir. Bulgu bu modellere özgü bir tuhaflık değil, ölçütün tanımından çıkan bir sonuçtur.

Dördüncüsü, açıklanamayan alarmların günlük döngüsü, bunların önemli bir bölümünün kataloglanmamış deprem değil kültürel gürültü olduğunu göstermektedir.

**Sınırlılıklar açıkça belirtilmelidir.** Sonuçlar tek istasyona (MANT) dayanmaktadır; istasyonlar arası değişkenlik ölçülmemiştir ve alan yazında bu değişkenliğin makine öğrenmesi tabanlı seçicilerde arttığına dair bulgular bulunmaktadır. Ölçüm tek istasyonlu ve ilişkilendirme (association) öncesidir; işletimdeki bütün erken uyarı sistemleri yanlış alarmları birden çok istasyonun uyuşmasını şart koşarak bastırmaktadır, dolayısıyla buradaki oranlar bir ağ için üst sınırdır. Gecikme ayrıştırması yapılmamıştır; yalnızca model terimi bilinmektedir, çünkü bir tespit ancak pencerenin tamamı gözlendikten sonra bildirilebilmektedir. Son olarak sinir ağı çıkışları kalibre edilmiş olasılık değildir ve bu çalışmada kalibre edilmemiştir.

Rapor, kısa pencereli sınıflandırıcıların sürekli veride kullanılamayacağını göstermemektedir. Gösterdiği şey, **düzenlenmiş bir sınama kümesinde belirlenen çalışma noktasının sürekli veriye taşınamayacağı** ve tespit edicilerin yalnızca toplulaştırılmış bir ölçütle karşılaştırılmasının yanıltıcı olabileceğidir.
