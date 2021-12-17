# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 02:09:37 2021

@author: youle
"""
import pathlib

names = [f'{alg}/1019' for alg in ["MO-MFEA", "MO-MFEA-II"]]
names.extend([f'{alg}/1017' for alg in ["EMEA", "Island_Model", "NSGA-II"]])

for parent_path in names:

    p = pathlib.Path(parent_path)
    dirs = p.glob("*design/")

    for d in dirs:

        for p in d.iterdir():
            p.unlink()
