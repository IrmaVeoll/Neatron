using System.Collections.Generic;
using XorDemo.Model;

namespace XorDemo.ViewModels
{
    public sealed class WinnerViewModel : ViewModelBase
    {
        public WinnerViewModel(ParetoFrontPoint paretoFrontPoint, IReadOnlyList<float> evaluationResult, int run, int generation)
        {
            ParetoFrontPoint = paretoFrontPoint;
            EvaluationResult = evaluationResult;
            Run = run;
            Generation = generation;
        }
        
        public ParetoFrontPoint ParetoFrontPoint { get; }
        
        public IReadOnlyList<float> EvaluationResult { get; }
        
        public int Run { get; }
        
        public int Generation { get; }
    }
}