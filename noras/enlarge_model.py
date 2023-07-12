import os
from rmgpy.rmg.main import RMG, RMG_Memory, log_conditions
import rmgpy.chemkin
from rmgpy.rmg.model import ReactionModel, CoreEdgeReactionModel


# 1. Use minimal example for RMG job 

# In[21]:


#specify input file and output directory

#CH4
output_dir = './output'
chemkin_file = './CH4/chem_annotated.inp'
species_dict = './CH4/species_dictionary.txt'
tran_file = './CH4/tran.dat'

#minimal
chemkin_file = './minimal/chem_annotated.inp'
chemkin_file_edge = './minimal/chem_edge_annotated.inp'
species_dict = './minimal/species_dictionary.txt'
species_dict_edge = './minimal/species_edge_dictionary.txt'


#load the chemkin file
species, reactions = rmgpy.chemkin.load_chemkin_file(chemkin_file, dictionary_path=species_dict)

species_edge, reactions_edge = rmgpy.chemkin.load_chemkin_file(chemkin_file_edge, dictionary_path=species_dict_edge)


# In[22]:


#initiate ReactionModel for core
model=ReactionModel()

#give it our species and reactions, this will be our "core"
model.species, model.reactions = species, reactions 


# In[47]:


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
        


# In[48]:


#initiate ReactionModel for edge
model_edge=ReactionModel()

#this will be our "edge"
model_edge.species, model_edge.reactions = species_edge_new , reactions_edge_new  

#initiate model for surface
model_surface=ReactionModel()

#this will be our surface
model_surface.species, model_surface.reactions = surface_species , surface_reactions 


# In[57]:


#combine into a CoreEdgeReactionModel

core_edge_model = CoreEdgeReactionModel(core=model, edge=model_edge, surface=model_surface)


#now enlarge
# core_edge_model.enlarge(model_surface.species[0])
core_edge_model.enlarge(react_edge=True)

