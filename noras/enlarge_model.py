#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import numpy as np
from rmgpy.rmg.main import RMG, RMG_Memory, log_conditions
import rmgpy.chemkin
from rmgpy.rmg.model import ReactionModel, CoreEdgeReactionModel
from rmgpy import settings
from rmgpy.data.rmg import RMGDatabase
from ipython.kinetics_library_to_training_tools import *


# 1. Use minimal example for RMG job 

#specify input file and output directory
full_path='/Users/nora/Code/projects/gpu_hackathon_2023'

#CH4
#output_dir = full_path+'/output'
#chemkin_file = full_path+'/CH4/chem_annotated.inp'
#species_dict = full_path+'/CH4/species_dictionary.txt'
#tran_file = full_path+'/CH4/tran.dat'

#minimal
chemkin_file = full_path+'/minimal/chem_annotated.inp'
chemkin_file_edge = full_path+'/minimal/chem_edge_annotated.inp'
species_dict = full_path+'/minimal/species_dictionary.txt'
species_dict_edge = full_path+'/minimal/species_edge_dictionary.txt'


#load the chemkin file
species, reactions = rmgpy.chemkin.load_chemkin_file(chemkin_file, dictionary_path=species_dict)

species_edge, reactions_edge = rmgpy.chemkin.load_chemkin_file(chemkin_file_edge, dictionary_path=species_dict_edge)


#initiate ReactionModel for core
model=ReactionModel()

#give it our species and reactions, this will be our "core"
model.species, model.reactions = species, reactions 


#let's make our edge and surface

#I found out that C5H9(93) has only participates in only one reaction in the edge. Let's make that species and that reaction our surface species

#pick out the species
surface_species=[]
species_edge_new = []
for ind, spc in enumerate(species_edge):
    if spc.index==93:
        #make this our surface species
        surface_species.append(spc)
    else:
        species_edge_new.append(spc)
        
#pick out the reaction
surface_reactions=[]
reactions_edge_new = []
for ind, rxn in enumerate(reactions_edge):
    if (surface_species[0] in rxn.products) or  (surface_species[0] in rxn.reactants):
        surface_reactions.append(rxn)
    else:
        reactions_edge_new.append(rxn)
        


#initiate ReactionModel for edge
model_edge=ReactionModel()

#this will be our "edge"
model_edge.species, model_edge.reactions = species_edge_new , reactions_edge_new  

#initiate model for surface
model_surface=ReactionModel()

#this will be our surface
model_surface.species, model_surface.reactions = surface_species , surface_reactions 


#have to load database in here somewhere#load database
families='default'
#set libraries to load reactions from; set to None to load all libraries
libraries =None

thermo_libraries = ['primaryThermoLibrary']

database = RMGDatabase()
database.load(
    path = settings['database.directory'],
    thermo_libraries = thermo_libraries,  # Can add others if necessary
    kinetics_families = families,
    reaction_libraries = libraries,
    kinetics_depositories = ['training'],
)

for family in database.kinetics.families.values():
        if not family.auto_generated:
            family.add_rules_from_training(thermo_database=database.thermo)
            family.fill_rules_by_averaging_up(verbose=False)

#combine into a CoreEdgeReactionModel

core_edge_model = CoreEdgeReactionModel(core=model, edge=model_edge)# surface=model_surface)

#Array of flags indicating whether a species should react unimolecularly in the enlarge step
unimolecular_react_array=[ True for i in range(len(species))]
#Array of flags indicating whether two species are above the bimolecular reaction threshold
bimolecular_react_array=np.full((len(species), len(species)), True, dtype=bool)
#len(species)=27. This is num_old_core_species in enlarge()

#now enlarge
#core_edge_model.enlarge(model_surface.species[0])
core_edge_model.enlarge(react_edge=True, unimolecular_react=unimolecular_react_array,bimolecular_react=bimolecular_react_array)

print('all done, you are truly the best')
