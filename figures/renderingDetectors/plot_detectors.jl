using SolidStateDetectors
using Plots

# Lista dei tuoi file JSON
files = [
    "BeGe_section.json", 
    "Coax_section.json", 
    "IC_section.json", 
    "PPC_section.json"
]

println("Librerie caricate. Inizio la generazione dei rendering puliti e ruotati...")

for file in files
    println("\nElaborazione di: ", file)
    
    # Carica la configurazione del rivelatore
    sim = Simulation{Float64}(file)
    
    # Genera il grafico con le impostazioni estetiche avanzate
    p = plot(sim.detector, 
        size = (1000, 1000),  # Aumentata leggermente la risoluzione per la tesi
        
        # RIMUOVE GLI ASSI E LA GRIGLIA
        framestyle = :axes,   
        grid = false,
        showaxis = false,
        ticks = false,
        
        # RUOTA LA TELECAMERA
        # Il primo numero (20) è l'azimuth: ruota l'oggetto a destra/sinistra.
        # Il secondo numero (25) è l'elevazione: alza la telecamera per guardare dall'alto verso il basso.
        # Se lo spaccato non è ancora perfettamente frontale, prova a cambiare 160 in 140, 180 o 200.
        camera = (20, 25),    
        
        title = "",           
        legend = false        
    )
    
    nome_immagine = replace(file, ".json" => "_frontale.png")
    
    # Salva l'immagine con sfondo trasparente (ottimo per presentazioni e tesi)
    savefig(p, nome_immagine)
    
    println("✅ Salvato: ", nome_immagine)
end

println("\nTutti i rendering sono stati generati con successo!")