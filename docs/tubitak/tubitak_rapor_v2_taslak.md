# TASLAK v2 — yeniden yapılandırma önerisi

> Bu dosya mevcut raporun yerini almaz; **yeni kurgunun şeklini ve tonunu**
> göstermek için yazılmıştır. Tümüyle yeni olan bölümler tam metin, mevcut
> bölümler ise yalnızca "ne değişiyor" notu hâlindedir. Sayılar
> `docs/experiment_ponly_2026-08-22.md` ve
> `docs/related_work_pwave_detection.md` dosyalarından alınmıştır.
>
> **Kurgu ilkesi değişmiyor:** BULGULAR ölçüm bildirir, TARTIŞMA yorum yapar.

---

## Kurgudaki temel değişiklik

Raporun ana savı artık bir dedektörün başarım değeri değil, **o değerin neye
göre ölçüldüğünün sonucu belirlediğidir.** Bu sav önceki sürümde tek bir
gösterime dayanıyordu; şimdi birbirinden bağımsız **üç** gösterime dayanmaktadır:

1. **Koşullu taban ile çoğunluk sınıfı tabanı arasındaki fark.** Görünürdeki
   0,491'lik katkı, koşullu tabana göre +0,0859'a inmektedir.
2. **Tabanın kendisinin yanlış olabilmesi.** ROC-AUC yalnızca monotonik
   sıralamayı ölçtüğünden, U biçimli bir ilişkiyi göremez: aynı tekil skaler
   üzerinde karar ağacı 0,7461'e ulaşırken ROC-AUC 0,6447 bildirmektedir.
3. **Negatif rejiminin sonucu kaydırması.** Aynı model ve aynı pozitiflerle,
   yalnızca negatiflerin seçim yöntemi değiştirilerek ROC-AUC 0,82 ile 0,91
   arasında değişmektedir.

Buna ikinci ve ayrı bir eksen eklenmektedir: **pencerenin içinde gerçekte ne
olduğu.** 6 s pencerelerin %28,8'i S varışını içermekte olup bu, hiçbir ölçütte
görünmeyen bir kurulum kusurudur.

Tüm bu düzeltmelerden sonra ayakta kalan olumlu sonuç ise şudur: **dedektör
dalga biçimi karakterini okumaktadır** — ve bu, iddia edilmek yerine
ölçülmüştür.

---

# 2. LİTERATÜR ÖZETİ

**Değişiklik:** 2.1–2.4 korunmaktadır. Yeni bir alt bölüm eklenmektedir.

## 2.5 Değerlendirme Protokollerinin Karşılaştırılması *(YENİ)*

Derin öğrenmeyle P dalgası tespiti yazınında bildirilen başarım değerleri 0,69
ile 0,98 arasında değişmektedir. Bu yayılım, mimariden çok **değerlendirme
protokolüne** bağlıdır. Aşağıdaki çizelge, başlık değerinden önce protokolü
karşılaştırmaktadır.

**Çizelge 2.1.** P dalgası tespiti yazınında değerlendirme protokolleri.

| | TransQuake (2021) | NZ edge CNN (2026) | CWT + YOLO (2025) | Bu çalışma |
|---|---|---|---|---|
| Görev | ikili: pencerede P var mı | **üç sınıf**: P / S / gürültü | ikili: spektrogram P mi | ikili: pencerede P var mı |
| Pencere | **50 s** | 2 s | görüntü başına | **3,4 s** (1,4 s P sonrası) |
| Negatifler | FilterPicker yanlış işaretlemeleri | **aynı 90 s kayıttan**, ±2 s dışlama | aynı istasyon kaydından | 3 sa önceki taranmış sakin pencereler, 482.898 olaylık katalogla ±300 s denetimli |
| Bölümleme | zamansal | **rastgele 70/15/15** | **istasyonlar arası** | **istasyon-ayrık** |
| Test dengesi | **~11:1** | dengeli | dengeli | dengeli |
| Büyüklük | artçılar | **M ≥ 3,0** | 3° içinde | **M ≥ 2,0** (medyan 2,3), ≤ 56 km |
| **Koşullu taban** | **bildirilmemiştir** | **bildirilmemiştir** | **bildirilmemiştir** | **0,6679** |
| Başlıca değer | P 0,740 / **R 0,685** | accuracy **%97,12** | P 0,934 / **R 0,942** | **ROC-AUC 0,8712**, R 0,638 |

Negatiflerin seçimini denetleyen iki çalışma recall 0,64–0,69 aralığında;
denetlemeyen ikisi 0,94–0,98 aralığındadır.

TransQuake pencere uzunluğu ablasyonunda 20–50 s aralığını taramış ve
"metriklerin, özellikle F1'in, pencere uzunluğu arttıkça iyileştiğini, bunun da
P dalgası dışındaki bilginin tespite katkıda bulunduğunu gösterdiğini"
bildirmiştir. Aynı çalışma, "farklı episantr uzaklıkları göz önüne alındığında
yalnızca tam bir P dalgası içeren sabit bir zaman penceresi belirlemenin mümkün
olmadığını" da kaydetmektedir.

NZ edge CNN çalışması, zamansal bölümlemenin "farklı dönemler arasında gözlenen
sapmalar nedeniyle başarısız olduğunu" ve bu nedenle rastgele bölümlemeye
geçildiğini bildirmektedir. Aynı çalışmada gürültü kesitleri, P ve S
kesitleriyle **aynı 90 s kayıttan** yalnızca ±2 s dışlama ile çekilmektedir.
Çalışmanın gömülü donanım katkısı (~38 bin parametre, 7 ms altı çıkarım) bu
karşılaştırmanın kapsamı dışındadır.

---

# 3. GEREÇ VE YÖNTEM

**Değişiklik:** 3.1, 3.2, 3.6, 3.7, 3.9 korunmaktadır. 3.3 genişlemekte, 3.4
yeni eklenmekte, 3.5 ve 3.8 genişlemektedir.

> **Düzeltilecek:** Mevcut 3.2, STFT parametrelerini `n_fft = 256, hop = 64`
> (`img` boyutu 3 × 129 × 10) olarak bildirmektedir. Başlıca sonuçların alındığı
> küme ise `n_fft = 64, hop = 16` (3 × 33 × 38) ile üretilmiştir. İki geometri
> de tanımlanmalı ve hangi sonucun hangisinden geldiği belirtilmelidir.

## 3.3 Varış Sabitleme *(genişletiliyor)*

**Eklenen:** Üretim yalnızca P fazlarını hesaplamaktadır
(`PHASES = ["p","P","Pg","Pn"]`). S varışının pencere içine düşüp düşmediği
üretim aşamasında denetlenmemektedir. Aynı (uzaklık, derinlik) çiftleri
üzerinde iasp91 ile hesaplandığında 6 s pencereler için:

**Çizelge 3.x.** 6 s penceresinde S varışının konumu.

| Uzaklık | Pencere | S pencere içinde | Medyan S−P |
|---|---|---|---|
| 0–25 km | 10.647 | **%99,3** | 2,56 s |
| 25–50 km | 35.074 | %15,5 | 5,11 s |
| 50–100 km | 9.847 | %0,0 | 6,70 s |
| **Tümü** | **55.568** | **%28,8** | **5,09 s** |

Tespit test bölümünde bu oran %32,5'tir.

## 3.4 Pencere Geometrisi: Yalnızca P *(YENİ)*

Pencere `[P − 2,0 s, P + 1,4 s]` olarak, toplam 3,4 s uzunlukta kesilmektedir.

Ön tampon pencere uzunluğuyla **ölçeklenmemektedir**. Üreticinin varsayılanı
`pencere/3` olup 3,4 s için 1,13 s vermektedir; kestirilen varışın 0,63 s
medyan mutlak sapması karşısında bu, başlangıcın pencere dışına düşmesine ve
tutulma oranının çökmesine yol açmaktadır. Ön tampon 2,0 s'de sabit
tutulduğunda kayıt düzeyinde tutulma **%96,4** olup 6 s yapılandırmasının
%96,3 değeriyle eşleşmektedir.

Üretilen 55.595 istasyon kaydının tamamı iasp91 ile yeniden denetlenmiştir:
**S varışının pencereye girdiği kayıt yoktur**; en küçük S−P 1,450 s, kesme
noktasına payı +0,050 s'dir.

Bu güvence **hız modeline görelidir, mutlak değildir.** S−P katalog
hiposantrından kestirildiğinden katalogun konum hatasını taşımaktadır (medyan
RMS kalıntısı 0,42 s):

**Çizelge 3.y.** S−P kestirim hatasına göre risk altındaki kayıtlar.

| S−P kestirimi şu kadar saparsa | S içerebilecek kayıt |
|---|---|
| 0,00 s | 0 |
| 0,30 s | 740 (%1,3) |
| 0,50 s | 1.445 (%2,6) |
| 0,63 s | 2.021 (%3,6) |

## 3.5 Negatif Seçimi *(genişletiliyor)*

**Eklenen:** Aynı pozitif pencereler üzerinde dört negatif rejimi
kurulmuştur. Dördü de aynı 35 test istasyonunu ve aynı 7.908 olay penceresini
paylaşmaktadır; değişen tek değişken negatif seçimidir.

| Rejim | Tanım |
|---|---|
| `matched` | Negatif genlik **dağılımı** pozitiflerinkini yansıtmaktadır |
| `band` | Havuzun %75–99 dilimi (yalnızca yüksek genlik) |
| `wideband` | %99 altındaki tüm havuz, dilimler arasında eşit yayılımlı |
| `natural` | Madencilik yok; havuzun kendi yoğunluğu |

**Genlik eşleştirmesinin gerekçesi.** %75–99 bandı her negatifin altına
pozitiflerde bulunmayan bir genlik tabanı koymaktadır. 6 s penceresinde bu
zararsızdır; yalnızca P içeren pencerede değildir (bkz. Bulgular).

## 3.8 Karşılaştırma Tabanları *(genişletiliyor)*

**Eklenen:** ROC-AUC yalnızca monotonik sıralamayı ölçmektedir. Bir tekil
istatistik ile sınıf arasındaki ilişki monotonik değilse, ROC-AUC o
istatistiğin öğrenilebilir kıldığı ayrımı **eksik bildirmektedir**. Bu nedenle
her taban iki biçimde hesaplanmaktadır:

- **Monotonik taban:** istatistiğin yönlendirilmiş ROC-AUC değeri.
- **Monotonik olmayan taban:** aynı tekil istatistik üzerinde, eğitim
  bölümünde uyarlanıp test bölümünde değerlendirilen 4 derinlikli karar ağacı.

İkisi arasındaki fark, kurulumun bir yapaylık taşıyıp taşımadığının
göstergesidir.

---

# 4. BULGULAR

**Kurgu:** Yalnızca P yapılandırması birincil, 6 s yapılandırması ise **önceki
kurulum** olarak sunulmaktadır. 6 s sonuçları geçerliliğini korumakta, ancak
farklı bir soruyu yanıtlamaktadır.

## 4.1 Koşullu Tabanların Ölçülmesi *(korunuyor, genişletiliyor)*

Her rejim için iki taban:

**Çizelge 4.1.** Negatif rejimine göre koşullu tabanlar (aynı pozitifler).

| Rejim | Monotonik | Monotonik olmayan | Fark |
|---|---|---|---|
| `matched` | 0,6679 | 0,6658 | −0,0021 |
| `band` | 0,6447 | **0,7461** | **+0,1015** |
| `wideband` | 0,7927 | 0,7845 | −0,0082 |
| `natural` | 0,7878 | 0,7795 | −0,0082 |

`band` rejiminde `P(olay | genlik)` desil bazında
0,67 · 0,41 · 0,32 · 0,29 · 0,30 · 0,33 · 0,30 · 0,50 · 0,88 · 1,00 değerlerini
almaktadır. Genlik eşleştirmesi sonrasında aynı dizi
0,40 · 0,38 · 0,39 · 0,35 · 0,41 · 0,42 · 0,45 · 0,53 · 0,70 · 0,96 olmaktadır.

Aynı denetim raporda taban bildirilen tüm kümelere uygulanmıştır; fark yalnızca
`band` rejiminde 0,02'yi aşmaktadır.

## 4.2 6 s Penceresinde S Varışının Katkısı *(YENİ)*

S varışından itibaren tüm örnekler sıfırlanarak mevcut ağırlıklarla yeniden
puanlanmıştır. Kuyruk sıfırlamanın kendisi de sinyal kaldırdığından, S
içermeyen pencerelere aynı uzunlukta kuyruk sıfırlama uygulanarak denetim
kurulmuştur.

**Çizelge 4.2.** S maskeleme ve süre eşleştirmeli denetim (recall, eşik 0,5).

| Küme | n | Maskesiz | Maskeli | Değişim |
|---|---|---|---|---|
| S içeren → S'den itibaren sıfırlanmış | 2.567 | 0,9747 | 0,9459 | −0,0288 |
| S içermeyen → dokunulmamış | 5.339 | 0,9358 | 0,9358 | +0,0000 |
| S içermeyen → aynı uzunlukta kuyruk sıfırlanmış | 5.339 | 0,9358 | 0,9002 | −0,0356 |

## 4.3 Yalnızca P: Alan İçi Başarım *(YENİ — birincil sonuç)*

**Çizelge 4.3.** Genlik eşleştirilmiş kümede yapılandırmalara göre başarım
(taban 0,6679, n = 15.816).

| Kol | Tohum başına | Topluluk | Katkı | Kazanılan pay |
|---|---|---|---|---|
| 1B | 0,8673 / 0,8709 / 0,8671 | 0,8712 | +0,2033 | %61,1 |
| 2B | 0,8605 / 0,8610 / 0,8544 | 0,8602 | +0,1923 | %57,9 |
| **Birleştirme** | 0,8730 / 0,8746 / 0,8737 | **0,8762** | +0,2083 | **%62,7** |

Karşılaştırma için 6 s yapılandırması (taban 0,9049): 1B 0,9896 (%89,1),
2B 0,9882 (%87,6), birleştirme 0,9908 (%90,3).

## 4.4 Negatif Rejimleri Arası Aktarım *(YENİ)*

**Çizelge 4.4.** Kazanılan pay: eğitim rejimi × değerlendirme rejimi.

| Eğitim | Kol | `matched` | `band` | `wideband` | `natural` |
|---|---|---|---|---|---|
| `matched` | 1B | %61,1 | %62,9 | %11,7 | %13,6 |
| `matched` | 2B | %57,9 | %53,4 | %14,4 | %16,2 |
| `matched` | Birleştirme | %62,7 | %65,4 | %14,4 | %16,0 |
| `natural` | 1B | %46,2 | %36,3 | %19,0 | %20,7 |
| `natural` | 2B | %33,9 | %12,1 | %20,9 | %23,1 |
| `natural` | Birleştirme | %43,0 | %30,1 | %22,8 | **%25,1** |

`natural` rejiminde birleştirilmiş modelin yanlış alarm sayısı, `natural`
eğitiminde 157, `matched` eğitiminde 317'dir; recall her ikisinde ~0,633'tür.

Recall bu çizelgede yer almamaktadır: pozitifler tüm rejimlerde özdeş
olduğundan recall yapısal olarak değişememektedir.

## 4.5 Genlik Sabit Tutulduğunda Ayrım *(YENİ)*

Genlik desilleri içinde ROC-AUC hesaplanmıştır. Dar desillerde genlik en çok
2,5 kat değişmektedir.

**Çizelge 4.5.** Genlik deseli içinde ROC-AUC (birleştirme, `matched`).

| Desil | n | Genlik aralığı | Genişlik | Desil içi AUC |
|---|---|---|---|---|
| 1 | 1.582 | 0,00 – 0,98 | 508,7× | 0,5664 |
| 2 | 1.581 | 0,98 – 1,51 | 1,5× | **0,6298** |
| 3 | 1.582 | 1,51 – 2,15 | 1,4× | **0,7090** |
| 4 | 1.581 | 2,15 – 3,01 | 1,4× | **0,7781** |
| 5 | 1.582 | 3,01 – 4,34 | 1,4× | **0,8013** |
| 6 | 1.581 | 4,34 – 6,43 | 1,5× | **0,8578** |
| 7 | 1.582 | 6,43 – 10,72 | 1,7× | **0,9274** |
| 8 | 1.581 | 10,72 – 21,85 | 2,0× | **0,9689** |
| 9 | 1.582 | 21,85 – 65,78 | 3,0× | 0,9902 |
| 10 | 1.582 | 65,78 – 34.794 | 528,9× | 0,9793 |

Kalın yazılan 7 desilde genlik en çok 2,5 kat değişmektedir; bu desillerin
medyanı **0,8013**'tür. 1., 9. ve 10. desillerde genlik 3–530 kat
değiştiğinden bu desiller kanıt oluşturmamaktadır.

`natural` kümesinde aynı ölçüm dar desil medyanı 0,7167 vermektedir.

## 4.6 İşletim Zarfı *(güncelleniyor — yalnızca P)*

Genel recall **0,6380**.

**Çizelge 4.6.** Kaynak parametrelerine göre recall.

| Uzaklık | n | Recall | | log SNR | n | Recall |
|---|---|---|---|---|---|---|
| 0–25 km | 1.587 | 0,6616 | | −2,0 – 0,72 | 2.629 | 0,4690 |
| 25–50 km | 5.044 | 0,6316 | | 0,72 – 3,42 | 4.261 | 0,7024 |
| 50–100 km | 1.274 | 0,6334 | | 3,42 – 6,13 | 921 | 0,8230 |
| | | | | > 6,13 | 55 | 0,9455 |

6 s yapılandırmasında aynı uzaklık dilimleri 0,9773 / 0,9417 / 0,9388
vermektedir.

Kaçırılan ve bulunan olayların medyanları: log SNR 0,722'ye karşılık 1,728;
uzaklık 39,33 km'ye karşılık 38,77 km.

## 4.7 Kalibrasyon ve Eşik *(güncelleniyor — yalnızca P)*

**Çizelge 4.7.** Kalibrasyon ölçütleri (n = 15.816).

| | ECE | MCE | Brier |
|---|---|---|---|
| Kalibrasyon öncesi | **0,0484** | 0,3009 | **0,1386** |
| Kalibrasyon sonrası (T = 0,6008) | 0,0765 | 0,1418 | 0,1444 |

**Çizelge 4.8.** Eşik seçenekleri.

| Eşik | MCC | Recall | Precision | Yanlış alarm |
|---|---|---|---|---|
| 0,50 | 0,6270 | 0,6383 | 0,9357 | 347 |
| 0,77 (MCC enbüyük) | 0,6368 | 0,6085 | 0,9715 | 141 |
| 0,90 | 0,6326 | 0,5885 | 0,9843 | 74 |

## 4.8 Önceki Kurulum: 6 s Penceresi *(mevcut 4.2–4.9 buraya taşınıyor)*

Mevcut 4.2, 4.3, 4.5, 4.6, 4.8, 4.9 bölümleri **önceki kurulum** başlığı
altında, kısaltılarak korunmaktadır.

## 4.9 Korpuslar Arası Genelleme *(kısaltılıyor)*

STEAD değerlendirmesi 6 s yapılandırmasıyla yapılmıştır; yalnızca P
yapılandırması bu korpusta henüz çalıştırılmamıştır. Bu nedenle burada
yalnızca özet olarak verilmektedir: katalog sabitli model eşleşmiş STEAD
kümesinde 0,9971 (taban 0,9752), tam aralıkta 0,9693 (taban 0,9531)
vermektedir. **Yalnızca P yapılandırmasının korpuslar arası sınanması
tamamlanmamıştır ve Bölüm 5.7'de sonraki adım olarak verilmektedir.**

---

# 5. TARTIŞMA

## 5.1 Karşılaştırma Tabanının Seçimi Sonucu Belirlemektedir *(genişletiliyor)*

**Mevcut metin korunmakta**, üzerine iki gösterim eklenmektedir.

**Taban da yanlış olabilir.** Koşullu taban, çoğunluk sınıfı tabanına göre
büyük bir düzeltmedir; ancak tabanın kendisi ROC-AUC ile ölçüldüğünde yalnızca
monotonik ilişkileri görmektedir. `band` rejiminde `P(olay | genlik)` U
biçimlidir — en sessiz desilin %67'si olaydır — ve ROC-AUC bunu 0,6447 olarak
bildirmektedir. Aynı tekil skaler üzerindeki karar ağacı 0,7461'e
ulaşmaktadır. Aradaki 0,1015, modelin görünürdeki payının önemli bir
bölümüdür. **Bir tabanın doğru olması, doğru istatistikle hesaplanmış olmasına
bağlıdır.**

**Değerlendirme kümesinin seçimi de sonucu kaydırmaktadır.** Bölüm 4.4'te aynı
model, aynı pozitiflerle ve yalnızca negatiflerin seçim yöntemi değiştirilerek
0,82 ile 0,91 arasında ROC-AUC vermektedir. Kazanılan pay bakımından aralık
%12 ile %65 arasındadır. Bu, mimari bir farkın on katından büyüktür.

Bu üç gösterim aynı yöne işaret etmektedir: **bildirilen sayı, ölçüm
protokolünün bir işlevidir.** Bölüm 2.5'teki çizelge aynı örüntünün yazında da
görüldüğünü göstermektedir.

## 5.2 Dedektör Dalga Biçimi Karakterini Okumaktadır *(YENİ)*

Bir tabanı geçmek, modelin genlikten başka bir şey kullandığını
**kanıtlamamaktadır**. Taban tekil bir skalerin ROC-AUC değeridir ve yalnızca
sıralamayı ölçmektedir; aynı skalerin daha iyi biçimlendirilmiş bir işlevi de
tabanı geçebilir. Bölüm 4.1 bu farkın bu veri kümesinde 0,1015 AUC ettiğini
göstermektedir.

Bölüm 4.5 genliği sabit tutarak bu ayrımı yapmaktadır. Genliğin en çok 2,5 kat
değiştiği 7 desilde model 0,63–0,97 arasında ROC-AUC almaktadır. **Genliğin
hiçbir işlevi bu değerleri üretemez.** Dedektörün dalga biçimi karakterini
okuduğu bu ölçümle gösterilmektedir.

Desil içi ROC-AUC genlikle birlikte tekdüze artmaktadır (0,63'ten 0,97'ye).
Bu, Bölüm 4.6'daki işletim zarfının SNR bağımlılığını bütünüyle bağımsız bir
hesaptan doğrulamaktadır.

Ölçümün **desteklemediği** iki nokta ayrıca kaydedilmelidir. En sessiz desil
508 kat genişliktedir; oradaki düşük değer hiçbir sonuç doğurmamaktadır.
Ayrıca `natural` eğitimli modelin dalga biçimini daha iyi okuduğu
söylenemez: aynı test kümesinde dar desil medyanları 0,7167 ve 0,7175'tir.

## 5.3 Eğitim Dağılımı ile Değerlendirme Dağılımı Ayrı Kararlardır *(YENİ)*

Genlik eşleştirilmiş küme bir **ölçüm aracıdır**, bir eğitim reçetesi değildir.
Amacı genliği bilgisiz kılarak geriye kalanı ölçmektir.

Bölüm 4.4 aktarımın **bakışımsız** olduğunu göstermektedir: `natural` eğitimli
model `matched` üzerinde %46,2, `matched` eğitimli model `natural` üzerinde
%13,6 pay kazanmaktadır. Geniş dağılımda eğitilen model her iki yöne de
aktarılabilmektedir; dar dağılımda eğitilen model aktarılamamaktadır.

Ancak eğitim dağılımı boşluğun tamamını kapatmamaktadır. `natural` üzerinde
`natural` eğitimi payı %13,6'dan %20,7'ye çıkarmakta, yani 47,5 puanlık
boşluğun 7,1 puanını kapatmaktadır. Kalan bölüm bir eğitim uyumsuzluğu
değildir.

## 5.4 İşletim Sınırını SNR Belirlemektedir *(güçleniyor)*

6 s yapılandırmasında recall uzaklıkla değişmekteydi (0,9773 → 0,9388) ve bu
SNR'nin dolaylı yansıması olarak yorumlanmıştı. Yalnızca P yapılandırmasında
aynı dilimler 0,6616 / 0,6316 / 0,6334, yani **düz**dür; kaçırılan ve bulunan
olaylar arasındaki medyan uzaklık farkı 0,5 km'ye inmektedir.

S varlığı uzaklıkla neredeyse birebir örtüştüğünden (25 km içinde %99,3, 50 km
ötesinde %0), 6 s yapılandırmasındaki uzaklık etkisinin önemli bir bölümünün S
katkısı olduğu değerlendirilmektedir. SNR bağımlılığı ise korunmakta ve
Bölüm 4.5'teki desil içi ölçümle bağımsız olarak doğrulanmaktadır.

## 5.5 Konuşlandırılabilir İddia *(YENİ)*

Bildirilebilecek iddia ROC-AUC değil, **recall**'dır: M ≥ 2,0 olaylar için,
56 km yarıçap içinde, P varışından sonraki 1,4 s ile **recall 0,61–0,64**.

Kalibrasyon bu yapılandırmada başarımı **kötüleştirmektedir** (ECE 0,0484 →
0,0765); 6 s yapılandırmasındaki bulgunun tersidir ve bu yapılandırma
kalibrasyonsuz bildirilmelidir.

Karar gecikmesi tasarım gereği P varışından 1,4 s sonradır. Bu, yazındaki
2 s (NZ edge CNN) ve 50 s (TransQuake) değerleriyle doğrudan
karşılaştırılabilir tek büyüklüktür.

## 5.6 Sınırlılıklar *(güncelleniyor)*

Mevcut maddelere eklenecekler:

- **Yalnızca P güvencesi hız modeline görelidir.** En yakın varışın payı
  0,050 s'dir; 0,63 s'lik bir kestirim hatası 2.021 kaydı (%3,6) riske
  sokmaktadır.
- **Genlik eşleştirmesi havuz derinliğiyle sınırlıdır.** Olay `seq` std değeri
  5.439'a ulaşırken havuzdaki en yüksek gürültü penceresi 77,8'dir. Eşleştirme
  alt yarıda tutmakta (1,15–1,35×), üst çeyrekte bozulmaktadır (p99'da 14,9×).
- **Sürekli veride yanlış alarm oranı ölçülmemiştir.** Yazındaki her
  konuşlandırılmış sistemin bildirdiği büyüklük budur. BODT istasyonu eğitim
  bölümünde bulunduğundan kullanılamaz; test bölümünde 35 istasyon
  ayrılmıştır.
- **Korpuslar arası sınama yalnızca 6 s yapılandırması için yapılmıştır.**

## 5.7 Sonraki Adımlar *(güncelleniyor)*

1. Yalnızca P yapılandırmasının STEAD üzerinde korpuslar arası sınanması.
2. Test bölümündeki istasyonlardan sürekli veri ile yanlış alarm oranının
   doğrudan ölçülmesi.
3. PhaseNet ve GPD ile bu korpusun kendi ölçüt kümesinde karşılaştırma.
4. 2B kolunun STFT zaman çözünürlüğü: `hop = 16` 160 ms/kare vermektedir; P
   başlangıcı onlarca ms'de gerçekleşmektedir.
