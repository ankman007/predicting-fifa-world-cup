TOURNAMENT_STRENGTH = {
    "fifa world cup": 1.0,
    
    "uefa euro": 0.95,
    "copa américa": 0.90,
    "african cup of nations": 0.85,
    "afc asian cup": 0.80,
    "gold cup": 0.75,
    "oceania nations cup": 0.65,  
    
    "confederations cup": 0.85,
    
    "uefa nations league": 0.80,
    "concacaf nations league": 0.70,
    
    "fifa world cup qualification": 0.75,
    "uefa euro qualification": 0.70,
    "african cup of nations qualification": 0.65,
    "afc asian cup qualification": 0.60,
    "gold cup qualification": 0.55,
    
    "arab cup": 0.65,
    "waff championship": 0.55,       
    "eaff championship": 0.55,       
    "cafa nations cup": 0.55,       
    "gulf cup": 0.55,
    "asian games": 0.50,            
    "cosafa cup": 0.50,             
    "amílcar cabral cup": 0.45,     
    "cfu caribbean cup": 0.45,      
    
    "superclásico de las américas": 0.55, 
    "tournoi de france": 0.50,
    "kirin challenge cup": 0.45,
    "kirin cup": 0.45,
    "usa cup": 0.45,
    "dynasty cup": 0.45,
    "soccer ashes": 0.45,           
    "trans-tasman cup": 0.45,
    "nordic championship": 0.45,
    "copa lipton": 0.40,
    "king hassan ii tournament": 0.40,
    "lunar new year cup": 0.40,
    "korea cup": 0.40,
    "nehru cup": 0.35,
    "king's cup": 0.35,
    "merdeka tournament": 0.35,
    "simba tournament": 0.35,
    "joe robbie cup": 0.35,
    "united arab emirates friendship tournament": 0.35,
    "jordan international tournament": 0.35,
    "prime minister's cup": 0.30,
    "copa confraternidad": 0.30,
    "osn cup": 0.30,
    "canadian shield": 0.30,
    "al ain international cup": 0.30,
    
    "friendly": 0.30,
    "fifa series": 0.30             
}

MAJOR_TOURNAMENTS = {
    k for k, v in TOURNAMENT_STRENGTH.items() if v >= 0.70
}