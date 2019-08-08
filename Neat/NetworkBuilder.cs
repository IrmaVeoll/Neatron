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
            private readonly Dictionary<ConnectionNeurons, (int neuronMapId, int neuronConnectionIndex)> _connectionGeneToNeuronMapId;
            private readonly Dictionary<int, (int neuronMapId, int layerId)> _neuronInfo;
            private readonly List<int> _hiddenLayerBounds = new List<int>();

            public NetworkMap(int connectionsCapacityHint, int neuronsCapacityHint)
            {
                _connectionGeneToNeuronMapId = new Dictionary<ConnectionNeurons, (int, int)>(connectionsCapacityHint);
                _neuronInfo = new Dictionary<int, (int neuronMapId, int layerId)>(neuronsCapacityHint);
            }

            public static NetworkMap CopyMaps(NetworkMap networkMap)
            {
                return new NetworkMap(networkMap);
            }

            private NetworkMap(NetworkMap networkMap)
            {
                _connectionGeneToNeuronMapId = new Dictionary<ConnectionNeurons, (int, int)>(networkMap._connectionGeneToNeuronMapId);
                _neuronInfo = new Dictionary<int, (int, int)>(networkMap._neuronInfo);
                IsDisconnected = networkMap.IsDisconnected;
                _hiddenLayerBounds = networkMap._hiddenLayerBounds;
            }

            public bool Contains(in ConnectionGene connectionGene) =>
                _connectionGeneToNeuronMapId.ContainsKey(connectionGene.ConnectionNeurons);

            public int ConnectionCount => _connectionGeneToNeuronMapId.Count;

            public void AddConnection(in ConnectionGene connectionGene, (int neuronMapId, int neuronConnectionIndex, int layerId) neuronInfo)
            {
                _connectionGeneToNeuronMapId.Add(connectionGene.ConnectionNeurons, (neuronInfo.neuronMapId, neuronInfo.neuronConnectionIndex));
                _neuronInfo[connectionGene.SourceId] = (neuronInfo.neuronMapId, neuronInfo.layerId);
                IsDisconnected = IsDisconnected && !connectionGene.Source.IsInput;
            }

            public bool TryGetNetworkIdx(in ConnectionGene connectionGene, out (int neuronMapId, int neuronConnectionIndex) neuronInfo) =>
                _connectionGeneToNeuronMapId.TryGetValue(connectionGene.ConnectionNeurons, out neuronInfo);

            public bool TryGetNeuronIdx(int neuronId, out (int neuronMapId, int layerId) neuronInfo) => _neuronInfo.TryGetValue(neuronId, out neuronInfo);

            public bool Contains(int neuronId) => _neuronInfo.ContainsKey(neuronId);

            public void AddLayerBound(int neuronMapId) => _hiddenLayerBounds.Add(neuronMapId);

            public List<int> HiddenLayerBounds => _hiddenLayerBounds;

            public bool IsDisconnected { get; private set; } = true;
            public Dictionary<int, (int idx, int layerId)> NeuronInfo => _neuronInfo;
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

                    _networkMap.AddConnection(in outerLayersConnection, (sourceId, j++, 0)); // TODO

                    idx++;

                    yield return new NetworkConnection(outerLayersConnection.TargetId, outerLayersConnection.Weight);
                }
            }

            for (var i = 0; i < _networkConnections.Length; i++)
            {
                _networkConnections[i] = GetOuterLayersConnections(i).ToArray();
            }

            Network = GetNetwork();
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

            Network = GetNetwork();
            Debug.Assert(IsValid());
        }

        internal INetwork Network { get; private set; }

        internal NetworkTopology NetworkTopology => new NetworkTopology(
            _neatChromosome.InputCount,
            _neatChromosome.OutputCount,
            _networkMap.HiddenLayerBounds,
            _networkConnections);

        internal int ConnectionCount => _networkMap.ConnectionCount;

        internal bool IsNetworkDisconnected => _networkMap.IsDisconnected;

        public void Encode()
        {
            for (var i = 0; i < _neatChromosome.Count; i++)
            {
                var connectionGene = _neatChromosome[i];
                if (_networkMap.TryGetNetworkIdx(in connectionGene, out var neuronInfo))
                {
                    _neatChromosome[i] =
                        new ConnectionGene(connectionGene.ConnectionNeurons,
                            _networkConnections[neuronInfo.neuronMapId][neuronInfo.neuronConnectionIndex].Weight,
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
                _networkMap.AddConnection(in connectionGene, (connectionGene.SourceId, outConnections.Length, 0));
                // No layers rebuilding needed
                Network = GetNetwork();
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

        private bool TryMapHiddenLayerConnection(IList<List<ConnectionGene>> layers, in ConnectionGene connectionGene, int currentLayerSize, int currentLayer)
        {
            if (_networkMap.Contains(in connectionGene))
            {
                return false;
            }

            if (!connectionGene.Source.IsHidden)
            {
                var sourceId = connectionGene.SourceId;
                layers[connectionGene.SourceId].Add(connectionGene);
                _networkMap.AddConnection(in connectionGene, (sourceId, layers[sourceId].Count - 1, currentLayer));
                return false;
            }

            if (_networkMap.TryGetNeuronIdx(connectionGene.SourceId, out var sourceNeuronInfo))
            {
                var sourceNeuronMapId = sourceNeuronInfo.neuronMapId;
                var sourceLayerNumber = sourceNeuronInfo.layerId;

                if (_networkType == FeedForward)
                {
                    int targetLayerNumber;
                    
                    // source neuron already has a connection to some layer. it has to lie in the next layer for FeedForward network
                    if (!_networkMap.TryGetNeuronIdx(connectionGene.TargetId, out var targetNeuronInfo))
                    {
                        if (connectionGene.Target.IsOutput)
                            targetLayerNumber = 0;
                        else
                            return false; // not processed hidden neuron. so it is in current layer or previous. both variants are bad
                    }
                    else
                        targetLayerNumber = targetNeuronInfo.layerId;

                    if (sourceLayerNumber <= targetLayerNumber || sourceLayerNumber - targetLayerNumber > 1)
                        return false;
                }

                layers[sourceNeuronMapId].Add(connectionGene);
                _networkMap.AddConnection(in connectionGene, (sourceNeuronMapId, layers[sourceNeuronMapId].Count - 1, currentLayer));
                return false;
            }


            layers.Add(new List<ConnectionGene> {connectionGene});
            _networkMap.AddConnection(in connectionGene, (layers.Count - 1, 0, currentLayer));
            return true;
        }

        private void AddNextLayer(IList<List<ConnectionGene>> layers, int previousLayerNeuronCount, int currentLayer)
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

                // for each connection in the last layer
                var layersCount = layers.Count;
                for (var i = layers.Count - previousLayerNeuronCount; i < layersCount; i++)
                {
                    var sourceId = layers[i][0].SourceId;

                    // get first connection to source neuron - other connections of this neuron are placed in a row (bucket)
                    var idx = _neatChromosome.FindFirstHiddenLayerConnectionIdx(sourceId);

                    if (idx < 0) // connection gene not found
                    {
                        continue;
                    }

                    Debug.Assert(hiddenLayersConnections[idx].TargetId == sourceId);

                    // all connections to this neurons belongs to previous layers
                    while (idx < hiddenLayersConnections.Count)
                    {
                        var hiddenLayerConnection = hiddenLayersConnections[idx];
                        if (hiddenLayerConnection.TargetId != sourceId) // end of connections bucket
                        {
                            break;
                        }

                        if (TryMapHiddenLayerConnection(layers, in hiddenLayerConnection, currentLayerNeuronCount, currentLayer))
                        {
                            currentLayerNeuronCount++;
                        }

                        idx++;
                    }
                }

                previousLayerNeuronCount = currentLayerNeuronCount;
                currentLayer++;
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

                    _networkMap.AddConnection(in outerLayersConnection, (sourceId, j++, 0)); // TODO 

                    idx++;

                    yield return outerLayersConnection;
                }
            }

            for (var i = 0; i < outerLayersNeuronCount; i++)
            {
                layers.Add(GetOuterLayersConnections(i).ToList());
            }

            var hiddenLayersConnections = _neatChromosome.HiddenLayersConnections;
            var currentLayer = 1;
            // adding last layer before effectors 
            for (var i = 0; i < hiddenLayersConnections.Count; i++)
            {
                var connectionGene = hiddenLayersConnections[i];
                if (!connectionGene.Target.IsOutput)
                {
                    // hidden layers are sorted as outer connections are always first
                    break;
                }

                _ = TryMapHiddenLayerConnection(layers, in connectionGene, layers.Count, currentLayer);
            }

            AddNextLayer(layers, layers.Count - outerLayersNeuronCount, ++currentLayer);
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
                        var res = _networkMap.TryGetNeuronIdx(connectionGene.TargetId, out var neuronInfo);
                        Debug.Assert(res);
                        connections[j] = new NetworkConnection(neuronInfo.neuronMapId, connectionGene.Weight); // todo
                    }
                }

                _networkConnections[i] = connections;
            }

            Network = GetNetwork();
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
                    .Select((cs, i) => new {connections = cs, sourceIdx = i})
                    .Any(s => s.connections.Any(c => c.TargetIdx == s.sourceIdx)))
                {
                    Debug.Fail("Self-loop detected in feedforward network.");
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

        private INetwork GetNetwork()
        {
            if(_networkType== FeedForward)
                return new Network(_networkConnections, _neatChromosome.InputCount, _neatChromosome.OutputCount, this);
            
            return new RecurrentNetwork(_networkConnections, _neatChromosome.InputCount, _neatChromosome.OutputCount, this);
        }
    }
}