using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using static System.Linq.Enumerable;
using static Neat.NetworkType;
using static Neat.NeuronGeneType;

using HiddenNeuronList = Neat.Utils.SortedList<Neat.NeuronGene, (int @in, int @out)>;

[assembly:InternalsVisibleTo("Tests")]
namespace Neat
{
    public sealed class NetworkParameters
    {
        public const int BiasCount = 1;

        private float _initialConnectionDensity = 1;

        private readonly List<NeuronGene> _inputs;
        private readonly List<NeuronGene> _outputs;
        private readonly VacantConnections _vacantConnections;

        public NetworkParameters(int sensorCount, int effectorCount, NetworkType networkType = Mixed)
        {
            if (sensorCount <= 0) throw new ArgumentOutOfRangeException(nameof(sensorCount));
            if (effectorCount <= 0) throw new ArgumentOutOfRangeException(nameof(effectorCount));

            SensorCount = sensorCount;
            EffectorCount = effectorCount;
            NetworkType = networkType;

            _inputs = new List<NeuronGene>(SensorCount + BiasCount)
            {
                new NeuronGene(Bias, 0)
            };
            _inputs.AddRange(Range(BiasCount, SensorCount)
                .Select(i => new NeuronGene(Sensor, i)));

            _outputs = new List<NeuronGene>(EffectorCount);
            _outputs.AddRange(Range(BiasCount + SensorCount, EffectorCount)
                .Select(i => new NeuronGene(Effector, i)));

            _vacantConnections = new VacantConnections(this);
        }

        public int SensorCount { get; }

        public int EffectorCount { get; }

        public NetworkType NetworkType { get; }

        public bool IsRecurrent => NetworkType != FeedForward;

        public float InitialConnectionDensity
        {
            get => _initialConnectionDensity;
            set
            {
                if (value <= 0 || value > 1)
                    throw new ArgumentOutOfRangeException(nameof(InitialConnectionDensity));
                
                _initialConnectionDensity = value;
            }
        }

        internal IReadOnlyList<NeuronGene> Inputs => _inputs;

        internal IReadOnlyList<NeuronGene> Outputs => _outputs;
        
        internal int GetVacantConnectionCount(NeatChromosome neatChromosome,
            HiddenNeuronList hiddenNeurons) =>
            _vacantConnections.GetVacantConnectionCount(neatChromosome, hiddenNeurons);

        internal ConnectionNeurons GetVacantConnectionNeurons(int vacantConnectionIdx, NeatChromosome neatChromosome,
            HiddenNeuronList hiddenNeurons) =>
            _vacantConnections.GetConnectionNeurons(vacantConnectionIdx, neatChromosome, hiddenNeurons);

    }
}