using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static Neat.NetworkType;

namespace Neat
{
    internal interface INeatChromosomeEncoder
    {
        void Encode();
    }

    internal sealed class NetworkBuilder : INeatChromosomeEncoder
    {
        private class NetworkMap
        {
            private readonly Dictionary<ConnectionNeurons, (int, int)> _connectionGeneToIdx;
            private readonly Dictionary<int, int> _neuronIdToIdx;
            private readonly List<int> _hiddenLayerBounds = new List<int>();

            public NetworkMap(int connectionsCapacityHint, int neuronsCapacityHint)
            {
                _connectionGeneToIdx = new Dictionary<ConnectionNeurons, (int, int)>(connectionsCapacityHint);
                _neuronIdToIdx = new Dictionary<int, int>(neuronsCapacityHint);
            }

            public static NetworkMap CopyMaps(NetworkMap networkMap)
            {
                return new NetworkMap(networkMap);
            }

            private NetworkMap(NetworkMap networkMap)
            {
                _connectionGeneToIdx = new Dictionary<ConnectionNeurons, (int, int)>(networkMap._connectionGeneToIdx);
                _neuronIdToIdx = new Dictionary<int, int>(networkMap._neuronIdToIdx);
                IsDisconnected = networkMap.IsDisconnected;
                _hiddenLayerBounds = networkMap._hiddenLayerBounds;
            }

            public bool Contains(in ConnectionGene connectionGene) =>
                _connectionGeneToIdx.ContainsKey(connectionGene.ConnectionNeurons);

            public int ConnectionCount => _connectionGeneToIdx.Count;

            public void AddConnection(in ConnectionGene connectionGene, (int i, int j) idx)
            {
                _connectionGeneToIdx.Add(connectionGene.ConnectionNeurons, idx);
                _neuronIdToIdx[connectionGene.SourceId] = idx.i;
                IsDisconnected = IsDisconnected && !connectionGene.Source.IsInput;
            }

            public bool TryGetNetworkIdx(in ConnectionGene connectionGene, out (int i, int j) idx) =>
                _connectionGeneToIdx.TryGetValue(connectionGene.ConnectionNeurons, out idx);

            public bool TryGetNeuronIdx(int neuronId, out int idx) => _neuronIdToIdx.TryGetValue(neuronId, out idx);

            public bool Contains(int neuronId) => _neuronIdToIdx.ContainsKey(neuronId);

            public void AddLayerBound(int idx) => _hiddenLayerBounds.Add(idx);

            public List<int> HiddenLayerBounds => _hiddenLayerBounds;

            public bool IsDisconnected { get; private set; } = true;
        }

        private readonly NeatChromosome _neatChromosome;
        private readonly NetworkType _networkType;

        private NetworkConnection[][] _networkConnections;
        private NetworkMap _networkMap;

        internal NetworkBuilder(NeatChromosome neatChromosome, NetworkType networkType)
        {
            _neatChromosome = neatChromosome;
            _networkType = networkType;

            _networkConnections = new NetworkConnection[_neatChromosome.OuterLayersNeuronCount][];
            _networkMap = new NetworkMap(_neatChromosome.Count,
                _neatChromosome.HiddenNeuronCount + _neatChromosome.OuterLayersNeuronCount);

            var idx = 0;
            IEnumerable<NetworkConnection> GetOuterLayersConnections(int sourceId)
            {
                var outerLayersConnections = _neatChromosome.OuterLayersConnections;
                var j = 0;
                while (idx < outerLayersConnections.Count)
                {
                    var outerLayersConnection = outerLayersConnections[idx];
                    if (outerLayersConnection.SourceId != sourceId)
                    {
                        yield break;
                    }

                    _networkMap.AddConnection(in outerLayersConnection, (sourceId, j++));

                    idx++;

                    yield return new NetworkConnection(outerLayersConnection.TargetId, outerLayersConnection.Weight);
                }
            }

            for (var i = 0; i < _networkConnections.Length; i++)
            {
                _networkConnections[i] = GetOuterLayersConnections(i).ToArray();
            }

            Network = new Network(_networkConnections, _neatChromosome.InputCount, _neatChromosome.OutputCount, this);
            Debug.Assert(IsValid());
        }

        internal NetworkBuilder(NetworkBuilder networkBuilder, NeatChromosome neatChromosome)
        {
            _neatChromosome = neatChromosome;
            
            _networkConnections = Copy(networkBuilder._networkConnections);
            for (var i = 0; i < _networkConnections.Length; i++)
            {
                _networkConnections[i] = Copy(_networkConnections[i]);
            }
            
            _networkMap = networkBuilder._networkMap;
            _networkType = networkBuilder._networkType;
            Network = new Network(_networkConnections, _neatChromosome.InputCount, _neatChromosome.OutputCount, this);
            Debug.Assert(IsValid());
        }

        internal Network Network { get; private set; }

        internal NetworkTopology NetworkTopology => new NetworkTopology(
            _neatChromosome.InputCount,
            _neatChromosome.OutputCount,
            _networkMap.HiddenLayerBounds,
            _networkConnections);

        internal int ConnectionCount => _networkMap.ConnectionCount;

        internal bool IsNetworkDisconnected => _networkMap.IsDisconnected;
        
        public void Encode()
        {
            if (_networkType == Recurrent)
            {
                throw new NotImplementedException(
                    "Looks you are trying to use backpropagation " + 
                    "on a recurrent network, but unfortunately it is not implemented yet. "
                                                                   + "Please be patient.");
            }

            for (var i = 0; i < _neatChromosome.Count; i++)
            {
                var connectionGene = _neatChromosome[i];
                if (_networkMap.TryGetNetworkIdx(in connectionGene, out var idx))
                {
                    _neatChromosome[i] =
                        new ConnectionGene(connectionGene.ConnectionNeurons, 
                            _networkConnections[idx.i][idx.j].Weight,
                            connectionGene.Id);
                }
            }
        }

        internal void UpdateWeight(in ConnectionGene connectionGene)
        {
            if (_networkMap.TryGetNetworkIdx(in connectionGene, out var idx))
            {
                var (i, j) = idx;
                _networkConnections[i][j] =
                    new NetworkConnection(_networkConnections[i][j].TargetIdx, connectionGene.Weight);
                Debug.Assert(IsValid());
            }
            
            Debug.Assert(IsValid());
        }

        internal void AddConnection(in ConnectionGene connectionGene)
        {
            // First of all let's process a couple of special cases:
            // 1. We've just added connection from/to hidden layer,
            if (!connectionGene.IsOuterLayersConnection &&
                // but output layer is disconnected from hidden layers (connection between introns added)
                !_neatChromosome.HasConnectedHiddenLayers)
            {
                // and it was definitely disconnected before this mutation,
                Debug.Assert(_networkConnections.Length == _neatChromosome.OuterLayersNeuronCount);
                // then we are lucky - even shallow coping of the _networkConnections is unnecessary.
                // Network has trivial structure - it contains only input and outputs layers
                Debug.Assert(IsValid());
                return;
            }

            // 2. We've added connection in outer layers, network's layers structure wasn't changed
            if (connectionGene.IsOuterLayersConnection)
            {
                // Create a shallow copy of the _networkConnections as we need to insert new connection somewhere
                _networkConnections = Copy(_networkConnections);
                // Get outgoing connections with the same source as added connection
                var outConnections = _networkConnections[connectionGene.SourceId];
                // Prepare new array for inserting new connection
                var newOutConnections = new NetworkConnection[outConnections.Length + 1];
                // Copy all old connections to the new array
                Copy(outConnections, newOutConnections);
                // Add new connection
                newOutConnections[outConnections.Length] =
                    new NetworkConnection(connectionGene.TargetId, connectionGene.Weight);
                // Insert new outgoing connections into _networkConnections at index corresponds to source
                _networkConnections[connectionGene.SourceId] = newOutConnections;
                // Create a copy of the _networkMap with the same layers list
                _networkMap = NetworkMap.CopyMaps(_networkMap);
                // Add new connection to _networkMap
                _networkMap.AddConnection(in connectionGene, (connectionGene.SourceId, outConnections.Length));
                // No layers rebuilding needed
                Network = new Network(_networkConnections, _neatChromosome.InputCount, _neatChromosome.OutputCount, this);
                Debug.Assert(IsValid());
                return;
            }

            // 3. New connection contains neurons, that is not mapped to network
            if (connectionGene.Source.IsHidden &&
                !_networkMap.Contains(connectionGene.SourceId) &&
                connectionGene.Target.IsHidden &&
                !_networkMap.Contains(connectionGene.TargetId))
            {
                Debug.Assert(IsValid());
                return;
            }

            RebuildNetwork();
            Debug.Assert(IsValid());
        }

        internal void RemoveConnection(in ConnectionGene connectionGene)
        {
            // Special cases:
            // 1. The network doesn't contain removed connection
            if (!_networkMap.Contains(in connectionGene))
            {
                Debug.Assert(IsValid());
                return;
            }

            // 2. We've just deleted connection from/to hidden layer,
            if (!connectionGene.IsOuterLayersConnection &&
                // and there are no hidden layers in chromosome or output layer is disconnected from hidden layers
                !_neatChromosome.HasConnectedHiddenLayers &&
                // but it was disconnected before this deletion,
                _networkConnections.Length == _neatChromosome.OuterLayersNeuronCount)
            {
                // then we are lucky - even shallow coping of the _networkConnections is unnecessary.
                // Network has trivial structure - it contains only input and outputs layers
                Debug.Assert(IsValid());
                return;
            }

            RebuildNetwork();
            Debug.Assert(IsValid());
        }

        internal void SplitConnection(in ConnectionGene connectionGene)
        {
            if (!_networkMap.Contains(in connectionGene))
            {
                Debug.Assert(IsValid());
                return;
            }

            RebuildNetwork();
            Debug.Assert(IsValid());
        }

        private bool TryMapHiddenLayerConnection(IList<List<ConnectionGene>> layers, in ConnectionGene connectionGene)
        {
            if (_networkMap.Contains(in connectionGene))
            {
                return false;
            }

            if (!connectionGene.Source.IsHidden)
            {
                var sourceId = connectionGene.SourceId;
                layers[connectionGene.SourceId].Add(connectionGene);
                _networkMap.AddConnection(in connectionGene, (sourceId, layers[sourceId].Count - 1));
                return false;
            }
            
            if (_networkMap.TryGetNeuronIdx(connectionGene.SourceId, out var i))
            {
                if (_networkType == FeedForward)
                {
                    return false;
                }
                
                layers[i].Add(connectionGene);
                _networkMap.AddConnection(in connectionGene, (i, layers[i].Count - 1));
                return false;
            }


            layers.Add(new List<ConnectionGene> { connectionGene });
            _networkMap.AddConnection(in connectionGene, (layers.Count - 1, 0));
            return true;
        }

        private void AddNextLayer(IList<List<ConnectionGene>> layers, int previousLayerNeuronCount)
        {
            while (true)
            {
                if (previousLayerNeuronCount == 0)
                {
                    return;
                }

                _networkMap.AddLayerBound(previousLayerNeuronCount);

                var hiddenLayersConnections = _neatChromosome.HiddenLayersConnections;
                var currentLayerNeuronCount = 0;

                // get all connection of last layer
                for (var i = layers.Count - previousLayerNeuronCount; i < layers.Count; i++)
                {
                    var sourceId = layers[i][0].SourceId;

                    // get first connection with current neuron - other connections of this neuron are placed in a row (bucket)
                    var idx = _neatChromosome.FindFirstHiddenLayerConnectionIdx(sourceId);

                    if (idx < 0) // connection gene not found
                    {
                        continue;
                    }

                    Debug.Assert(hiddenLayersConnections[idx].TargetId == sourceId);

                    while (idx < hiddenLayersConnections.Count)
                    {
                        var hiddenLayerConnection = hiddenLayersConnections[idx];
                        if (hiddenLayerConnection.TargetId != sourceId) // end of connections bucket
                        {
                            break;
                        }

                        if (TryMapHiddenLayerConnection(layers, in hiddenLayerConnection))
                        {
                            currentLayerNeuronCount++;
                        }

                        idx++;
                    }
                }

                previousLayerNeuronCount = currentLayerNeuronCount;
            }
        }

        private IReadOnlyList<IReadOnlyList<ConnectionGene>> BuildLayers()
        {
            var outerLayersNeuronCount = _neatChromosome.OuterLayersNeuronCount;
            _networkMap = new NetworkMap(_neatChromosome.Count,
                _neatChromosome.HiddenNeuronCount + _neatChromosome.OuterLayersNeuronCount);
            var layers = new List<List<ConnectionGene>>(_neatChromosome.HiddenNeuronCount + 2);

            var idx = 0;
            IEnumerable<ConnectionGene> GetOuterLayersConnections(int sourceId)
            {
                var outerLayersConnections = _neatChromosome.OuterLayersConnections;
                var j = 0;
                while (idx < outerLayersConnections.Count)
                {
                    var outerLayersConnection = outerLayersConnections[idx];
                    if (outerLayersConnection.SourceId != sourceId)
                    {
                        yield break;
                    }

                    _networkMap.AddConnection(in outerLayersConnection, (sourceId, j++));

                    idx++;

                    yield return outerLayersConnection;
                }
            }

            for (var i = 0; i < outerLayersNeuronCount; i++)
            {
                layers.Add(GetOuterLayersConnections(i).ToList());
            }

            var hiddenLayersConnections = _neatChromosome.HiddenLayersConnections;

            // looking for last hidden layer 
            for (var i = 0; i < hiddenLayersConnections.Count; i++)
            {
                var connectionGene = hiddenLayersConnections[i];
                if (!connectionGene.Target.IsOutput)
                {
                    // hidden layers are sorted as outer connections are always first
                    break;
                }

                _ = TryMapHiddenLayerConnection(layers, in connectionGene);
            }

            AddNextLayer(layers, layers.Count - outerLayersNeuronCount);
            return layers;
        }

        private void RebuildNetwork()
        {
            var layers = BuildLayers();

            _networkConnections = new NetworkConnection[layers.Count][];

            for (var i = 0; i < layers.Count; i++)
            {
                var connections = new NetworkConnection[layers[i].Count];
                for (var j = 0; j < connections.Length; j++)
                {
                    var connectionGene = layers[i][j];
                    if (connectionGene.Target.IsOutput)
                    {
                        connections[j] = new NetworkConnection(connectionGene.TargetId, connectionGene.Weight);
                    }
                    else
                    {
                        var res = _networkMap.TryGetNeuronIdx(connectionGene.TargetId, out var idx);
                        Debug.Assert(res);
                        connections[j] = new NetworkConnection(idx, connectionGene.Weight);
                    }
                }

                _networkConnections[i] = connections;
            }

            Network = new Network(_networkConnections, _neatChromosome.InputCount, _neatChromosome.OutputCount, this);
        }

        private static T[] Copy<T>(T[] source)
        {
            var copy = new T[source.Length];
            Copy(source, copy);
            return copy;
        }

        private static void Copy<T>(T[] source, T[] destination)
        {
            Array.Copy(source, destination, Math.Min(source.Length, destination.Length));
        }

        private bool IsValid()
        {
            if (_networkConnections.Any(t => t.GroupBy(x => x.TargetIdx).Any(g => g.Count() > 1)))
            {
                Debug.Fail("Network contains duplicate connections.");
                return false;
            }

            if (_neatChromosome.HasConnectedHiddenLayers &&
                _networkConnections.Length == _neatChromosome.OuterLayersNeuronCount)
            {
                Debug.Fail("Network must contain hidden layers.");
                return false;
            }

            if (_neatChromosome.OuterLayersConnections.Any(c => !_networkMap.Contains(c)))
            {
                Debug.Fail("Unmapped outer layers connections found.");
                return false;
            }

            if (_networkConnections.SelectMany(c => c).Count() > _neatChromosome.Count)
            {
                Debug.Fail("Invalid chromosome decoding.");
                return false;
            }

            if (_networkConnections.SelectMany(c => c).Count() != ConnectionCount)
            {
                Debug.Fail("Invalid ConnectionCount.");
                return false;
            }
            
            if (!_networkConnections.SelectMany(c => c).All(c =>
                _neatChromosome.OuterLayersConnections.Concat(_neatChromosome.HiddenLayersConnections)
                    .Any(cg => cg.Weight.Equals(c.Weight))))
            {
                Debug.Fail("Invalid chromosome decoding.");
                return false;
            }
            
            if (_networkMap.IsDisconnected !=
                !_networkConnections.Take(_neatChromosome.InputCount).SelectMany(_ => _).Any())
            {
                Debug.Fail("Invalid IsDisconnected property value.");
                return false;
            }

            if (_networkType == FeedForward)
            {
                if (_networkConnections
                    .Select((cs, i) => new { connections = cs, sourceIdx = i })
                    .Any(s => s.connections.Any(c => c.TargetIdx == s.sourceIdx)))
                {
                    Debug.Fail("Self-loop detected in feedforward network.");
                    return false;
                }

                var connectionsToPrevLayer = _networkConnections
                    .Select((cs, i) => new { connections = cs, sourceIdx = i })
                    .Where(s => s.connections.Any(c => c.TargetIdx > s.sourceIdx
                                                       && c.TargetIdx > _neatChromosome.OuterLayersNeuronCount
                                                       && s.sourceIdx > _neatChromosome.OuterLayersNeuronCount));

                if (connectionsToPrevLayer.Any())
                {
                    Debug.Fail("Back connection detected in feedforward network.");
                    return false;
                }

                // looking for hidden layers to detect in-layer bounds links 
                for (var layerIndex = 1; layerIndex < NetworkTopology.LayerRanges.Count - 1; layerIndex++)
                {
                    var layerRange = NetworkTopology.LayerRanges[layerIndex];
                    var layerStartId = layerRange.Item1;
                    var layerEndId = layerRange.Item1 + layerRange.Item2 - 1;

                    for (var neuronId = layerStartId; neuronId <= layerEndId; neuronId++)
                    {
                        // connection between neurons of same level 
                        foreach (var withinLink in NetworkTopology.Links[neuronId].Where(l => l.TargetIdx >= layerStartId && l.TargetIdx <= layerEndId))
                        {
                            // they can connect each other, only if only one of them is connected to next layer
                            // in other words - if they can be splitted into separate layers in topology
                            var curNeuronHasLinksToNexLevel = NetworkTopology.Links[neuronId].Any(link => link.TargetIdx < layerStartId);
                            var connectedNeuronHasLinksToNexLevel = NetworkTopology.Links[withinLink.TargetIdx].Any(link => link.TargetIdx < layerStartId);
                            if (curNeuronHasLinksToNexLevel && connectedNeuronHasLinksToNexLevel)
                            {
                                Debug.Fail("connection between neurons of same level detected in feedforward network.");
                            }
                        }
                    }
                }
            }

            return true;
        }
    }
}