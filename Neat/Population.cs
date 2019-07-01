using System;
using static Neat.Genome;

namespace Neat
{
    public sealed class Population
    {
        private readonly NetworkParameters _networkParameters;

        public Population(NetworkParameters networkParameters, 
            ReproductionParameters reproductionParameters,
            bool relaxedNeatGenesComparison = true)
        {
            _networkParameters = networkParameters ?? throw new ArgumentNullException(nameof(networkParameters));
            
            Replicator = new Replicator(reproductionParameters ??
                                        throw new ArgumentNullException(nameof(reproductionParameters)));
            InnovationTracker = new InnovationTracker(NetworkParameters.BiasCount + 
                                                      _networkParameters.SensorCount +
                                                      _networkParameters.EffectorCount);

            RelaxedNeatGenesComparison = relaxedNeatGenesComparison;
        }

        public bool RelaxedNeatGenesComparison { get; }

        public InnovationTracker InnovationTracker { get; }
        
        public Replicator Replicator { get; }

        public Genome CreateInitialGenome()
        {
            return new Genome(_networkParameters, InnovationTracker, RelaxedNeatGenesComparison);
        }
    }
}