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
        size = (1000, 1000),  # Risoluzione elevata per la tesi
        
        # MOSTRA SOLO GLI ASSI CENTRALI SENZA RIQUADRO
        framestyle = :origin, 
        grid = false,
        
        # Rimuove etichette, numeri e tacche dagli assi
        xticks = false,
        yticks = false,
        xlabel = "",
        ylabel = "",
        
        # RUOTA LA TELECAMERA
        # Il primo numero (20) è l'azimuth: ruota l'oggetto a destra/sinistra.
        # Il secondo numero (25) è l'elevazione: alza la telecamera per guardare dall'alto verso il basso.
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