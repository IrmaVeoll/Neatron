using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace Neat
{
    internal sealed class NeatChromosome : IReadOnlyList<ConnectionGene>
    {
        private enum MaxInnovationIdState
        {
            Invalid = -2,
            Uninitialized = -1
        }

        private static readonly Comparer<ConnectionGene> ConnectionGeneSourceComparer =
            Comparer<ConnectionGene>.Create((c1, c2) =>
            {
                var res = c1.SourceId.CompareTo(c2.SourceId);
                return res != 0 ? res : c1.TargetId.CompareTo(c2.TargetId);
            });
        
        private static readonly Comparer<ConnectionGene> ConnectionGeneTargetComparer =
            Comparer<ConnectionGene>.Create((c1, c2) =>
            {
                var res = c1.TargetId.CompareTo(c2.TargetId);
                return res != 0 ? res : c1.SourceId.CompareTo(c2.SourceId);
            });
        
        private readonly List<ConnectionGene> _outerLayersConnections;
        private readonly List<ConnectionGene> _hiddenLayersConnections;

        private int _maxInnovationId = (int) MaxInnovationIdState.Uninitialized;

        public NeatChromosome(int inputCount, int outputCount, int capacity)
        {
            _outerLayersConnections = new List<ConnectionGene>(capacity);
            _hiddenLayersConnections = new List<ConnectionGene>();
            
            InputCount = inputCount;
            OutputCount = outputCount;
            OuterLayersNeuronCount = inputCount + outputCount;
        }

        public NeatChromosome(NeatChromosome neatChromosome)
        {
            _outerLayersConnections = new List<ConnectionGene>(neatChromosome.OuterLayersConnections);
            _hiddenLayersConnections = new List<ConnectionGene>(neatChromosome.HiddenLayersConnections);

            InputCount = neatChromosome.InputCount;
            OutputCount = neatChromosome.OutputCount;
            OuterLayersNeuronCount = neatChromosome.OuterLayersNeuronCount;
            HiddenNeuronCount = neatChromosome.HiddenNeuronCount;

            _maxInnovationId = neatChromosome._maxInnovationId;
        }
        
        internal int InputCount { get; }

        internal int OutputCount { get; }

        internal int OuterLayersNeuronCount { get; }

        internal bool HasConnectedHiddenLayers =>
            _hiddenLayersConnections.Count > 0 && _hiddenLayersConnections[0].Target.IsOutput;

        internal int HiddenNeuronCount { get; set; }

        internal int MaxInnovationId
        {
            get
            {
                if (_maxInnovationId < 0)
                {
                    _maxInnovationId = (int) MaxInnovationIdState.Uninitialized;
                    
                    for (var i = 0; i < _outerLayersConnections.Count; i++)
                    {
                        var id = _outerLayersConnections[i].Id;
                        if (_maxInnovationId < id)
                        {
                            _maxInnovationId = id;
                        }
                    }
                    
                    for (var i = 0; i < _hiddenLayersConnections.Count; i++)
                    {
                        var id = _hiddenLayersConnections[i].Id;
                        if (_maxInnovationId < id)
                        {
                            _maxInnovationId = id;
                        }
                    }
                }

                return _maxInnovationId;
            }
        }

        public IEnumerator<ConnectionGene> GetEnumerator()
        {
            for (var i = 0; i < Count; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public void Add(in ConnectionGene item)
        {
            var (list, idx) = Search(in item);
            if (idx < 0)
            {
                idx = ~idx;
            }
            list.Insert(idx, item);

            if (_maxInnovationId != (int) MaxInnovationIdState.Invalid && _maxInnovationId < item.Id)
            {
                _maxInnovationId = item.Id;
            }
        }
        
        internal bool TryFindConnectionGene(in ConnectionGene connectionGene, out ConnectionGene foundConnectionGene)
        {
            var (list, idx) = Search(connectionGene);

            if (idx < 0)
            {
                foundConnectionGene = default;
                return false;
            }

            foundConnectionGene = list[idx];
            return true;
        }
        
        internal int FindFirstHiddenLayerConnectionIdx(int targetId)
        {
            var lo = 0;
            var hi = _hiddenLayersConnections.Count - 1;

            while (lo <= hi)
            {
                var i = lo + ((hi - lo) >> 1);
                var res = _hiddenLayersConnections[i].TargetId.CompareTo(targetId);
                if (res == 0)
                {
                    while (--i >= 0 && _hiddenLayersConnections[i].TargetId == targetId) { }
                    return i + 1;
                }

                if (res < 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }

        public bool Contains(in ConnectionGene item) => Search(in item).index >= 0;

        public int Count => OuterLayersConnections.Count + HiddenLayersConnections.Count;

        public IReadOnlyList<ConnectionGene> OuterLayersConnections => _outerLayersConnections;

        public IReadOnlyList<ConnectionGene> HiddenLayersConnections => _hiddenLayersConnections;

        internal void Remove(int idx, int id)
        {
            RemoveAt(idx);
            
            if (id == _maxInnovationId)
            {
                _maxInnovationId = (int) MaxInnovationIdState.Invalid;
            }
        }

        private void RemoveAt(int index)
        {
            var (list, idx) = MapIndex(index);
            list.RemoveAt(idx);
        }

        public ConnectionGene this[int index]
        {
            get
            {
                var (list, idx) = MapIndex(index);
                return list[idx];
            }
            set 
            {
                var (list, idx) = MapIndex(index);
                Debug.Assert(Equals(list[idx].ConnectionNeurons, value.ConnectionNeurons));
                list[idx] = value;
            }
        }

        private (IList<ConnectionGene> list, int index) MapIndex(int index) => index < _outerLayersConnections.Count
            ? (_outerLayersConnections, index)
            : (_hiddenLayersConnections, index - _outerLayersConnections.Count);

        private (IList<ConnectionGene> list, int index) Search(in ConnectionGene item)
        {
            var (list, comparer) = item.IsOuterLayersConnection
                ? (_outerLayersConnections, ConnectionGeneSourceComparer)
                : 
                (_hiddenLayersConnections, ConnectionGeneTargetComparer);
            
            var index = list.BinarySearch(item, comparer);
            
            return (list, index);
        }
    }
}