RxCUIs = """151871
216755
1999309
316965
317541
5552
105602
213282
213283
213284
1999314
1999316
564113
573195
573196
573197
1999310
1999315
366743
366459
1999311
197797
200342
200343
200344
283475
1999308
316067
316068
316069
316070
331560
1999307
370657
370658
1160113
1160114
1151131
1151133
1166387
1166388
1174066
1174067
1999312
1999313
"""

RxCUI_list = RxCUIs.splitlines()
print(RxCUI_list)




import xml.etree.ElementTree as ET
tree = ET.parse('EDA/folate.xml')
root = tree.getroot()

root.tag
root.attrib

[elem.tag for elem in root.iter()]

print(ET.tostring(root, encoding='utf8').decode('utf8'))

for cui in root.iter('RXCUI'):
  print(cui.attrib)

Folate_CUIs = [cui.text for cui in root.iter('RXCUI')]

len(Folate_CUIs)