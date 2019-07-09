using System.Collections.Generic;
using HiddenNeuronList = Neat.Utils.SortedList<Neat.NeuronGene, (int @in, int @out)>;

namespace Neat
{
    internal sealed class VacantConnections
    {
        private readonly bool _isRecurrent;
        
        private readonly IReadOnlyList<NeuronGene> _inputs;
        private readonly IReadOnlyList<NeuronGene> _outputs;

        private readonly int _maxOuterLayersConnectionCount;

        internal VacantConnections(NetworkParameters networkParameters)
        {
            _isRecurrent = networkParameters.IsRecurrent;
            
            _inputs = networkParameters.Inputs;
            _outputs = networkParameters.Outputs;

            _maxOuterLayersConnectionCount =
                _inputs.Count * _outputs.Count + (_isRecurrent ? _outputs.Count * _outputs.Count : 0);
        }

        internal int GetVacantConnectionCount(NeatChromosome neatChromosome, HiddenNeuronList hiddenNeurons)
        {
            var maxInputToHiddenCount = _inputs.Count * hiddenNeurons.Count;
            var maxHiddenToOutputCount = hiddenNeurons.Count * _outputs.Count;
            var maxOutputToHiddenCount = _isRecurrent ? maxHiddenToOutputCount : 0;
            var maxHiddenToHiddenCount = hiddenNeurons.Count * (_isRecurrent
                                             ? hiddenNeurons.Count
                                             : hiddenNeurons.Count - 1);
            return _maxOuterLayersConnectionCount +
                   maxInputToHiddenCount +
                   maxHiddenToOutputCount +
                   maxOutputToHiddenCount +
                   maxHiddenToHiddenCount - neatChromosome.Count;
        }

        public ConnectionNeurons GetConnectionNeurons(int vacantConnectionIdx, 
            NeatChromosome neatChromosome, HiddenNeuronList hiddenNeurons)
        {
            var outerLayersVacantConnectionCount =
                _maxOuterLayersConnectionCount - neatChromosome.OuterLayersConnections.Count;

            return vacantConnectionIdx < outerLayersVacantConnectionCount
                ? GetOuterLayersVacantConnectionNeurons(vacantConnectionIdx, neatChromosome.OuterLayersConnections)
                : GetHiddenLayersVacantConnectionNeurons(vacantConnectionIdx - outerLayersVacantConnectionCount,
                    neatChromosome, hiddenNeurons);
        }

        private ConnectionNeurons GetHiddenLayersVacantConnectionNeurons(int vacantConnectionIdx,
            NeatChromosome neatChromosome, HiddenNeuronList hiddenNeurons)
        {
            NeuronGene GetSourceNeuronGene(int sourceNeuronIdx, int targetNeuronIdx)
            {
                if (sourceNeuronIdx < _inputs.Count)
                {
                    return _inputs[sourceNeuronIdx];
                }

                if (_isRecurrent)
                {
                    return sourceNeuronIdx < _inputs.Count + _outputs.Count
                        ? _outputs[sourceNeuronIdx - _inputs.Count]
                        : hiddenNeurons.Keys[sourceNeuronIdx - _inputs.Count - _outputs.Count];
                }

                //feedforward
                var sourceHiddenNeuronIdx = sourceNeuronIdx - _inputs.Count;
                return sourceHiddenNeuronIdx < targetNeuronIdx
                    ? hiddenNeurons.Keys[sourceHiddenNeuronIdx]
                    : hiddenNeurons.Keys[sourceHiddenNeuronIdx + 1];
            }
            
            var maxInConnectionCount = _inputs.Count + hiddenNeurons.Count + (_isRecurrent ? _outputs.Count : -1);
            
            var vacantConnectionSum = 0;
            int hiddenNeuronIdx;
            var inConnectionCount = 0;
            for (hiddenNeuronIdx = 0; hiddenNeuronIdx < hiddenNeurons.Count; hiddenNeuronIdx++)
            {
                inConnectionCount = hiddenNeurons.Values[hiddenNeuronIdx].@in;
                vacantConnectionSum += maxInConnectionCount - inConnectionCount;
                if (vacantConnectionSum > vacantConnectionIdx)
                {
                    break;
                }
            }
            
            var hiddenLayersConnections = neatChromosome.HiddenLayersConnections;
            
            // There are no enough incoming vacant connections in hidden neurons, search in hidden to output connections
            if (vacantConnectionSum <= vacantConnectionIdx)
            {
                return GetHiddenToOuterLayerVacantConnectionNeurons(vacantConnectionIdx - vacantConnectionSum,
                    hiddenLayersConnections, hiddenNeurons);
            }

            var hiddenNeuronVacantConnectionIdx =
                vacantConnectionIdx - (vacantConnectionSum - (maxInConnectionCount - inConnectionCount));
            
            var targetHiddenNeuron = hiddenNeurons.Keys[hiddenNeuronIdx];
                
            if (inConnectionCount == 0)
            {
                return new ConnectionNeurons(GetSourceNeuronGene(hiddenNeuronVacantConnectionIdx, hiddenNeuronIdx), targetHiddenNeuron);
            }
            
            var connectionIdx = neatChromosome.FindFirstHiddenLayerConnectionIdx(targetHiddenNeuron.Id);

            int connectionCount;
            var connection = hiddenLayersConnections[connectionIdx];
                
            for (connectionCount = 0;
                connection.Target == targetHiddenNeuron;
                connection = hiddenLayersConnections[connectionIdx])
            {
                if (!connection.Source.IsHidden)
                {
                    var vacantConnectionCount = connection.SourceId - connectionCount;
                    if (vacantConnectionCount > hiddenNeuronVacantConnectionIdx)
                    {
                        break;
                    }
                }
                else //connection source is hidden neuron
                {
                    var sourceNeuronIdx = hiddenNeurons.IndexOfKey(connection.Source);
                    var vacantConnectionCount = _inputs.Count + 
                                                (_isRecurrent ? _outputs.Count : 0) +
                                                sourceNeuronIdx - 
                                                (!_isRecurrent && sourceNeuronIdx >= hiddenNeuronIdx ? 1 : 0) -
                                                connectionCount;
                    if (vacantConnectionCount > hiddenNeuronVacantConnectionIdx)
                    {
                        break;
                    }
                }
                    
                connectionCount++;

                if (++connectionIdx == hiddenLayersConnections.Count)
                {
                    break;
                }
            }

            var sourceIdx = connectionCount + hiddenNeuronVacantConnectionIdx;
            return new ConnectionNeurons(GetSourceNeuronGene(sourceIdx, hiddenNeuronIdx), targetHiddenNeuron);
        }

        private ConnectionNeurons GetHiddenToOuterLayerVacantConnectionNeurons(int vacantConnectionIdx,
            IReadOnlyList<ConnectionGene> hiddenLayersConnections, HiddenNeuronList hiddenNeurons)
        {
            int connectionCount;
            var connection = hiddenLayersConnections[0];
            for (connectionCount = 0;
                connection.Target.IsOutput;
                connection = hiddenLayersConnections[connectionCount])
            {
                var vacantConnectionCount = (connection.TargetId - _inputs.Count) * hiddenNeurons.Count +
                                            hiddenNeurons.IndexOfKey(connection.Source) - connectionCount;
                if (vacantConnectionCount > vacantConnectionIdx)
                {
                    break;
                }

                if (++connectionCount == hiddenLayersConnections.Count)
                {
                    break;
                }
            }

            var connectionIdx = connectionCount + vacantConnectionIdx;
            var sourceIdx = connectionIdx % hiddenNeurons.Count;
            var targetIdx = connectionIdx / hiddenNeurons.Count;

            return new ConnectionNeurons(hiddenNeurons.Keys[sourceIdx], _outputs[targetIdx]);
        }

        private ConnectionNeurons GetOuterLayersVacantConnectionNeurons(int vacantConnectionIdx,
            IReadOnlyList<ConnectionGene> outerLayersConnections)
        {
            NeuronGene GetSourceNeuronGene(int neuronIdx)
            {
                return neuronIdx < _inputs.Count ? _inputs[neuronIdx] : _outputs[neuronIdx - _inputs.Count];
            }

            int connectionCount;
            for (connectionCount = 0;
                connectionCount < outerLayersConnections.Count;
                connectionCount++)
            {
                var connection = outerLayersConnections[connectionCount];
                var vacantConnectionCount =
                    connection.SourceId * _outputs.Count + (connection.TargetId - _inputs.Count) - connectionCount;
                if (vacantConnectionCount > vacantConnectionIdx)
                {
                    break;
                }
            }

            var connectionIdx = connectionCount + vacantConnectionIdx;
            var sourceIdx = connectionIdx / _outputs.Count;
            var targetIdx = connectionIdx % _outputs.Count;
            
            return new ConnectionNeurons(GetSourceNeuronGene(sourceIdx), _outputs[targetIdx]);
        }
    }
}