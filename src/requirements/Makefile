# ========================================
# Makefile for the Master Thesis of the MSE
#
# Author: Yannis Schmutz
# Created 03.03.2023
# ========================================

# *** Helpful infos ***
#
# target: pre-req1 pre-req2 pre-req3 ...
#    recipes
#
# MAKE SURE TO USE **TABS** before the recipe lists.
# Anything other than tabs will result in an error.
# *********************

requirements.txt: requirements.in
	python3.11 -m piptools compile --output-file ./requirements.txt requirements.in
