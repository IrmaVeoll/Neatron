using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Neat.Utils;
using static System.Math;
using static Neat.CrossoverType;
using static Neat.NetworkType;
using static Neat.NeuronGeneType;

using HiddenNeuronList = Neat.Utils.SortedList<Neat.NeuronGene, (int @in, int @out)>;

namespace Neat
{
    public sealed class Genome
    {
        private class ManhattanDistance : INeatChromosomesDistance
        {
            private float _sum;

            public void Prepare() => _sum = 0f;

            public void HandleMatchedGenes(in ConnectionGene cg1, in ConnectionGene cg2) =>
                _sum += Abs(cg1.Weight - cg2.Weight);

            public void HandleExcessGene(in ConnectionGene cg) => _sum += Abs(cg.Weight);

            public void HandleDisjointGene(in ConnectionGene cg) => _sum += Abs(cg.Weight);

            public float GetDistance() => (float) Sqrt(_sum);
        }

        private static readonly Comparer<NeuronGene> NeuronGeneComparer =
            Comparer<NeuronGene>.Create((n1, n2) => n1.Id.CompareTo(n2.Id));
        
        private static readonly INeatChromosomesDistance DefaultDistanceProvider = new ManhattanDistance();

        private readonly NetworkParameters _networkParameters;
        private readonly InnovationTracker _innovationTracker;
        private readonly bool _relaxedNeatGenesComparison;
        
        private readonly NeatChromosome _neatChromosome; 
        private readonly NetworkBuilder _networkBuilder;
        
        private HiddenNeuronList _hiddenNeurons = new HiddenNeuronList(NeuronGeneComparer);
        private bool _hiddenNeuronsNeedCopyOnWrite;

        internal Genome(NetworkParameters networkParameters, InnovationTracker innovationTracker,
            bool relaxedNeatGenesComparison)
        {
            _networkParameters = networkParameters;
            _innovationTracker = innovationTracker;
            _relaxedNeatGenesComparison = relaxedNeatGenesComparison;
            
            var inputs = networkParameters.Inputs;
            var outputs = networkParameters.Outputs;

            _neatChromosome =
                new NeatChromosome(inputs.Count, outputs.Count, (inputs.Count + outputs.Count) * outputs.Count);

            IEnumerable<ConnectionNeurons> GetAllowedConnections()
            {
                for (var i = 0; i < inputs.Count; i++)
                {
                    for (var j = 0; j < outputs.Count; j++)
                    {
                        if (networkParameters.InitialConnectionDensity > RandomSource.Next())
                        {
                            yield return new ConnectionNeurons(inputs[i], outputs[j]);
                        }
                    }
                }

                if (networkParameters.NetworkType == FeedForward)
                {
                    yield break;
                }
                
                for (var i = 0; i < outputs.Count; i++)
                {
                    for (var j = 0; j < outputs.Count; j++)
                    {
                        if (networkParameters.InitialConnectionDensity > RandomSource.Next())
                        {
                            yield return new ConnectionNeurons(outputs[i], outputs[j]);
                        }
                    }
                }
            }

            foreach (var connection in GetAllowedConnections())
            {
                AddConnectionInternal(connection, ConnectionGeneWeightGenerator.GetRandomUniform());
            }

            _networkBuilder = new NetworkBuilder(_neatChromosome, networkParameters.NetworkType);
            
            if (_neatChromosome.Count == 0)
            {
                AddConnection(RandomSource.Next(VacantConnectionCount));
            }

            Debug.Assert(IsValid());
        }

        private Genome(Genome genome)
        {
            _networkParameters = genome._networkParameters;
            _innovationTracker = genome._innovationTracker;
            _relaxedNeatGenesComparison = genome._relaxedNeatGenesComparison;

            _hiddenNeurons = genome._hiddenNeurons;
            _hiddenNeuronsNeedCopyOnWrite = true;
            
            _neatChromosome = new NeatChromosome(genome._neatChromosome);
            _networkBuilder = new NetworkBuilder(genome._networkBuilder, _neatChromosome);
            
            Debug.Assert(IsValid());
        }
        
        public INetwork Network => _networkBuilder.Network;

        public NetworkTopology NetworkTopology => _networkBuilder.NetworkTopology;

        public int HiddenLayersConnectionCount => _neatChromosome.HiddenLayersConnections.Count;

        public int NetworkConnectionCount => _networkBuilder.ConnectionCount;

        public bool IsNetworkDisconnected => _networkBuilder.IsNetworkDisconnected;

        public int HiddenNeuronCount => _hiddenNeurons.Count;

        internal IReadOnlyList<ConnectionGene> NeatChromosome => _neatChromosome;

        internal int VacantConnectionCount =>
            _networkParameters.GetVacantConnectionCount(_neatChromosome, _hiddenNeurons);

        public Genome Clone() => new Genome(this);

        public float GetNeatChromosomesDistance(Genome other) =>
            GetNeatChromosomesDistance(other, DefaultDistanceProvider);
        
        public float GetNeatChromosomesDistance(Genome other, INeatChromosomesDistance distanceProvider)
        {
            distanceProvider.Prepare();
                
            for (var i = 0; i < _neatChromosome.Count; i++)
            {
                var thisGenomeConnection = _neatChromosome[i];

                if (other._neatChromosome.TryFindConnectionGene(in thisGenomeConnection,
                        out var otherGenomeConnection) &&
                    (_relaxedNeatGenesComparison || otherGenomeConnection.Id == thisGenomeConnection.Id))
                {
                    distanceProvider.HandleMatchedGenes(in thisGenomeConnection, in otherGenomeConnection);
                }
                else
                {
                    if (thisGenomeConnection.Id > other._neatChromosome.MaxInnovationId)
                    {
                        distanceProvider.HandleExcessGene(in thisGenomeConnection);
                    }
                    else
                    {
                        distanceProvider.HandleDisjointGene(in thisGenomeConnection);
                    }
                }
            }
            
            for (var i = 0; i < other._neatChromosome.Count; i++)
            {
                var otherGenomeConnection = other._neatChromosome[i];

                if (_relaxedNeatGenesComparison && _neatChromosome.Contains(in otherGenomeConnection) ||
                    _neatChromosome.TryFindConnectionGene(in otherGenomeConnection, out var thisGenomeConnection) &&
                    thisGenomeConnection.Id == otherGenomeConnection.Id)
                {
                    continue;
                }

                if (otherGenomeConnection.Id > _neatChromosome.MaxInnovationId)
                {
                    distanceProvider.HandleExcessGene(in otherGenomeConnection);
                }
                else
                {
                    distanceProvider.HandleDisjointGene(in otherGenomeConnection);
                }
            }

            return distanceProvider.GetDistance();
        }
        
        internal Genome Mate(Genome worseGenome, CrossoverType crossoverType)
        {
            var p1 = RandomSource.Next(_neatChromosome.Count);
            var p2 = RandomSource.Next(_neatChromosome.Count);
            var swapGeneIndex1 = p1 <= p2 ? p1 : p2;
            var swapGeneIndex2 = p1 <= p2 ? p2 : p1;
            var startFromBest = RandomSource.NextBool();

            var adjustedCrossoverType = crossoverType == TwoPoints && p1 == p2 ? OnePoint : crossoverType;
            
            var child = Clone();
            
            for (var i = 0; i < _neatChromosome.Count; i++)
            {
                var thisGenomeConnection = _neatChromosome[i];

                if (!worseGenome._neatChromosome.TryFindConnectionGene(in thisGenomeConnection,
                        out var otherGenomeConnection) ||
                    (!_relaxedNeatGenesComparison && otherGenomeConnection.Id != thisGenomeConnection.Id))
                {
                    continue;
                }
                
                switch (adjustedCrossoverType)
                {
                    case Uniform:
                    {
                        if (RandomSource.NextBool())
                        {
                            child.UpdateWeight(in thisGenomeConnection, i, otherGenomeConnection.Weight);
                        }

                        break;
                    }
                    case ArithmeticRecombination:
                    {
                        child.UpdateWeight(in thisGenomeConnection, i,
                            (thisGenomeConnection.Weight + otherGenomeConnection.Weight) / 2);
                        break;
                    }

                    case OnePoint:
                    {
                        if (i >= swapGeneIndex1 && !startFromBest)
                        {
                            return child;
                        }

                        if (i < swapGeneIndex1 && startFromBest)
                        {
                            continue;
                        }
                        
                        child.UpdateWeight(in thisGenomeConnection, i, otherGenomeConnection.Weight);

                        break;
                    }

                    case TwoPoints:
                    {
                        if (i >= swapGeneIndex1 && i <= swapGeneIndex2)
                        {
                            if (startFromBest)
                            {
                                child.UpdateWeight(in thisGenomeConnection, i, otherGenomeConnection.Weight);
                            }
                        }
                        else
                        {
                            if (!startFromBest)
                            {
                                child.UpdateWeight(in thisGenomeConnection, i, otherGenomeConnection.Weight);
                            }
                        }

                        break;
                    }

                    default:
                        throw new ArgumentOutOfRangeException(nameof(crossoverType), crossoverType, null);
                }
            }

            return child;
        }

        internal void UpdateWeight(in ConnectionGene connectionGene, int connectionGeneIdx, float newWeight)
        {
            var newConnectionGene = new ConnectionGene(connectionGene.ConnectionNeurons,
                newWeight,
                connectionGene.Id);
            _neatChromosome[connectionGeneIdx] = newConnectionGene;
            _networkBuilder.UpdateWeight(in newConnectionGene);
            
            Debug.Assert(IsValid());
        }

        internal void AddConnection(int vacantConnectionIdx)
        {
            var connectionNeurons = _networkParameters.GetVacantConnectionNeurons(vacantConnectionIdx, _neatChromosome, _hiddenNeurons);

            var newConnection =
                AddConnectionInternal(connectionNeurons, ConnectionGeneWeightGenerator.GetRandomUniform());
            RegisterHiddenNeurons(in newConnection);
            
            _networkBuilder.AddConnection(in newConnection);
            
            Debug.Assert(IsValid());
        }

        internal void RemoveConnection(int connectionGeneIdx)
        {
            var connectionGene = _neatChromosome[connectionGeneIdx];
            _neatChromosome.Remove(connectionGeneIdx, connectionGene.Id);
            _networkBuilder.RemoveConnection(in connectionGene);

            UnregisterHiddenNeurons(in connectionGene);

            Debug.Assert(IsValid());
        }

        internal void SplitConnection(int connectionGeneIdx)
        {
            float Clamp(float value, float maxAbsValue)
            {
                if (value > maxAbsValue)
                    return maxAbsValue;
                if (value < -maxAbsValue)
                    return -maxAbsValue;
                return value;
            }

            (float w1, float w2) GetWeights(in ConnectionGene connectionGene, float maxAbsWeightValue)
            {
                var w = Abs(connectionGene.Weight);
                
                if (!(w > float.Epsilon))
                {
                    return (0, 0);
                }

                if (connectionGene.Source.IsBias)
                {
                    return (ConnectionGene.MaxAbsWeightValue * Sign(connectionGene.Weight),
                        Clamp(w * 1.1f, maxAbsWeightValue));
                }

                const double a = 0.0000127212307424923d;
                const double b = 0.000187725313497866f;
                const double c = 0.00173497083092954f;
                const double d = 0.015099962668533f;
                const double e = 0.0999436144686452f;

                var wSquared = w * w;

                var w1 = a * wSquared * wSquared * w + b * wSquared * wSquared + c * wSquared * w + d * wSquared + e * w;

                return (Clamp((float) w1, maxAbsWeightValue),
                    ConnectionGene.MaxAbsWeightValue * Sign(connectionGene.Weight));
            }
            
            var oldConnectionGene = _neatChromosome[connectionGeneIdx];
            var oldConnectionNeurons = oldConnectionGene.ConnectionNeurons;
            
            var newNeuronId = _innovationTracker.RegisterSplitConnection(in oldConnectionGene);
            var newNeuron = new NeuronGene(Hidden, newNeuronId);

            var weights = GetWeights(in oldConnectionGene, 1000000f);
            
            var connectedNeurons1 = new ConnectionNeurons(oldConnectionNeurons.Source, newNeuron);
            var connectionGene1 = AddConnectionInternal(connectedNeurons1, weights.w1);
            RegisterHiddenNeurons(in connectionGene1);
            
            var connectedNeurons2 = new ConnectionNeurons(newNeuron, oldConnectionNeurons.Target);
            var connectionGene2 = AddConnectionInternal(connectedNeurons2, weights.w2);
            RegisterHiddenNeurons(in connectionGene2);
            
            _neatChromosome.Remove(connectionGeneIdx, oldConnectionGene.Id);
            UnregisterHiddenNeurons(in oldConnectionGene);
            
            _networkBuilder.SplitConnection(in oldConnectionGene);
            
            Debug.Assert(IsValid());
        }

        private ConnectionGene AddConnectionInternal(ConnectionNeurons connectionNeurons, float weight)
        {
            var innovationId = _innovationTracker.RegisterAddConnection(connectionNeurons);
            var connection = new ConnectionGene(connectionNeurons, weight, innovationId);
            Debug.Assert(!_neatChromosome.Contains(in connection), 
                $"Connection {connection.ConnectionNeurons} has already been added.");
            _neatChromosome.Add(in connection);
            return connection;
        }

        private void RegisterHiddenNeurons(in ConnectionGene connectionGene)
        {
            void AddRef(NeuronGene neuronGene, int @in, int @out)
            {
                if (!neuronGene.IsHidden)
                {
                    return;
                }

                CopyHiddenNeuronList();

                if (_hiddenNeurons.TryGetValue(neuronGene, out var refCount))
                {
                    _hiddenNeurons[neuronGene] = (refCount.@in + @in, refCount.@out + @out);
                }
                else
                {
                    _hiddenNeurons[neuronGene] = (@in, @out);
                    _neatChromosome.HiddenNeuronCount++;
                }
            }
            
            AddRef(connectionGene.Source, 0, 1);
            AddRef(connectionGene.Target, 1, 0);
        }
        
        private void UnregisterHiddenNeurons(in ConnectionGene connectionGene)
        {
            void Release(NeuronGene neuronGene, int @in, int @out)
            {
                if (!neuronGene.IsHidden)
                {
                    return;
                }
                
                CopyHiddenNeuronList();
                
                var res = _hiddenNeurons.TryGetValue(neuronGene, out var refCount);
                Debug.Assert(res);
                refCount = (refCount.@in - @in, refCount.@out - @out);
                if (refCount.@in == 0 && refCount.@out == 0)
                {
                    _hiddenNeurons.Remove(neuronGene);
                    _neatChromosome.HiddenNeuronCount--;
                }
                else
                {
                    _hiddenNeurons[neuronGene] = refCount;
                }
            }
            
            Release(connectionGene.Source, 0, 1);
            Release(connectionGene.Target, 1, 0);
        }
        
        private void CopyHiddenNeuronList()
        {
            if (_hiddenNeuronsNeedCopyOnWrite)
            {
                _hiddenNeurons = new HiddenNeuronList(_hiddenNeurons);
                _hiddenNeuronsNeedCopyOnWrite = false;
            }
        }

        private bool IsValid()
        {
            var inputCount = _neatChromosome.InputCount;
            var outputCount = _neatChromosome.OutputCount;
            
            if (_neatChromosome.OuterLayersConnections.Any(c => c.Source.Id >= inputCount + outputCount || 
                                                                c.Target.Id >= inputCount + outputCount))
            {
                Debug.Fail("Chromosome contains a hidden layer connection in outer layers connections.");
                return false;
            }
            
            if (_neatChromosome.HiddenLayersConnections.Any(c => c.Source.Id < inputCount + outputCount && 
                                                                 c.Target.Id < inputCount + outputCount))
            {
                Debug.Fail("Chromosome contains an outer layer connection in hidden layers connections.");
                return false;
            }

            if (_neatChromosome.OuterLayersConnections.Union(_neatChromosome.HiddenLayersConnections)
                .Select(c => c.ConnectionNeurons).Any(c => c.TargetId < inputCount))
            {
                Debug.Fail("Some connections contain input neurons as a target.");
                return false;
            }

            if (_hiddenNeurons.Keys.Any(n => n.Id < inputCount + outputCount))
            {
                Debug.Fail("Hidden neuron list contains an outer layer neurons.");
                return false;
            }

            if (_neatChromosome.HiddenNeuronCount != _hiddenNeurons.Count)
            {
                Debug.Fail("Chromosome contains invalid neurons count.");
                return false;
            }

            if (_neatChromosome.HiddenLayersConnections.Any(c =>
                c.Source.IsHidden && !_hiddenNeurons.ContainsKey(c.Source) ||
                c.Target.IsHidden && !_hiddenNeurons.ContainsKey(c.Target)))
            {
                Debug.Fail("Chromosome contains connections with not registered hidden neurons.");
                return false;
            }

            if (_hiddenNeurons.Any(kv =>
                _neatChromosome.HiddenLayersConnections.Count(c => c.Source == kv.Key) +
                _neatChromosome.HiddenLayersConnections.Count(c => c.Target == kv.Key) !=
                kv.Value.@in + kv.Value.@out))
            {
                Debug.Fail("Invalid reference count in hidden neuron list.");
                return false;
            }

            if (_hiddenNeurons.Values.Any(count => count.@in < 1 && count.@out < 1))
            {
                Debug.Fail("Invalid reference count in hidden neuron list.");
                return false;
            }

            if (_hiddenNeurons.Keys.Any(n => !_neatChromosome.HiddenLayersConnections.Any(c =>
                c.Source.IsHidden && c.Source == n || c.Target.IsHidden && c.Target == n)))
            {
                Debug.Fail("Hidden neurons contain unreferenced neuron.");
                return false;
            }

            if (_neatChromosome.HiddenLayersConnections.Concat(_neatChromosome.OuterLayersConnections).Any() &&
                _neatChromosome.HiddenLayersConnections.Concat(_neatChromosome.OuterLayersConnections).Max(c => c.Id) !=
                _neatChromosome.MaxInnovationId)
            {
                Debug.Fail("Invalid max innovation id.");
                return false;
            }

            if (!_networkParameters.IsRecurrent && _neatChromosome.Any(cg => cg.Source == cg.Target))
            {
                Debug.Fail("Self-loop detected in feedforward network genome.");
                return false;
            }
            
            if (!_networkParameters.IsRecurrent && _neatChromosome.Any(cg => cg.Source.IsOutput))
            {
                Debug.Fail("Connection with with output neuron source detected in feedforward network genome.");
                return false;
            }

            return true;
        }
    }
}