# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:35:25 2023

@author: liyulin
"""
import matplotlib.pyplot as plt
from pyteomics import mzml, pepxml
from pyteomics import mgf, pepxml, mass, pylab_aux
import os
from urllib.request import urlopen, Request
import pylab

path_mzml = 'C:/Users/liyulin/Desktop/Transformer/test/sgsHuman_01/napedro_L120417_010_SW_Q1.mzML'

# path_mzml = 'C:/Users/liyulin/Desktop/Transformer/test/search_ehgines/LFQ_TTOF6600_DDA_QC_01.mzML'



path_pxml = 'C:/Users/liyulin/Desktop/Transformer/test/sgsHuman_01/napedro_L120417_010_SW_Q1.pepXML'
# path_pxml = 'C:/Users/liyulin/Desktop/Transformer/test/search_ehgines\msfragger/LFQ_TTOF6600_DDA_QC_01.pepXML'

# Read the spectrum data
spectra = mzml.MzML(path_mzml)

Psm = pepxml.read(path_pxml)
#%%
i = 80
spectrum = spectra[i]
psm = Psm[i]

print(spectrum['id'])
print(psm['spectrumNativeID'])


pylab.figure()
pylab_aux.annotate_spectrum(spectrum, psm['search_hit'][0]['peptide'],
    title='Annotated spectrum ' + psm['search_hit'][0]['peptide'],
    maxcharge=psm['assumed_charge'])

#%%
def fragments(peptide, types=('b', 'y'), maxcharge=1):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxharge`.
    """
    for i in range(1, len(peptide)):
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                if ion_type[0] in 'abc':
                    yield mass.fast_mass(
                            peptide[:i], ion_type=ion_type, charge=charge)
                else:
                    yield mass.fast_mass(
                            peptide[i:], ion_type=ion_type, charge=charge)
                    
fragment_list = list(fragments(psm['search_hit'][0]['peptide'], maxcharge=psm['assumed_charge']))
theor_spectrum = {'m/z array': fragment_list, 'intensity array': [spectrum['intensity array'].max()] * len(fragment_list)}

pylab.figure()
pylab.title('Theoretical and experimental spectra for ' + psm['search_hit'][0]['peptide'])
pylab_aux.plot_spectrum(spectrum, width=0.1, linewidth=2, edgecolor='black')
pylab_aux.plot_spectrum(theor_spectrum, width=0.1, edgecolor='red', alpha=0.7)
pylab.show()
#%%
from pyteomics import parser
forms = parser.isoforms('PSTADGEGDERPFTQAGLGADER', variable_mods={'p': ['T'], 'ox': ['P']})
for seq in forms: 
    print(seq)