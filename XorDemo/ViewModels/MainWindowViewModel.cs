using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reactive.Disposables;
using System.Threading;
using Avalonia.Threading;
using Neat;
using ReactiveUI;
using XorDemo.Model;

namespace XorDemo.ViewModels
{
    public sealed class MainWindowViewModel : ViewModelBase, ISupportsActivation
    {
        private const float AnimationSpeed = -0.03f;
        private const int PeriodicalUpdateTimeoutMilliseconds = 50;

        private static readonly PopulationParameters PopulationParameters =
            new PopulationParameters(200, 2);

        private static readonly NetworkParameters NetworkParameters =
            new NetworkParameters(2, 1, NetworkType.FeedForward)
            {
                InitialConnectionDensity = 0.9f
            };

        private static readonly ReproductionParameters ReproductionParameters =
            new ReproductionParameters
            {
                CrossoverType = CrossoverType.Uniform,

                WeightMutations = new WeightMutations
                {
                    OverallRouletteWheelShare = 80,
                    Mutations =
                    {
                        new WeightTweak
                        {
                            RouletteWheelShare = 100f,
                            Sigma = 0.1f
                        },
                        new WeightTweak
                        {
                            RouletteWheelShare = 20f,
                            ConnectionCount = 2,
                            Sigma = 0.01f
                        },
                        new WeightTweak
                        {
                            RouletteWheelShare = 1f,
                            ConnectionCount = 3,
                            Sigma = 0.008f
                        },
                        new WeightPerturb
                        {
                            RouletteWheelShare = 2f
                        }
                    }
                },
                AddConnectionRouletteWheelShare = 6f,
                RemoveConnectionRouletteWheelShare = 8f,
                SplitConnectionRouletteWheelShare = 4f
            };

        private readonly CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        private readonly EventWaitHandle _runSearchLoop = new ManualResetEvent(true);
        private Measurement _complexityMeasure;
        private Measurement _fitnessMeasure;
        private int _measureCount;
        private WinnerViewModel _leaderOfRun;

        private Timer _periodicalUpdate;
        private Thread _searchThread;

        public MainWindowViewModel()
        {
            FitnessGraph = new ObservableCollection<Measurement>();
            ComplexityGraph = new ObservableCollection<Measurement>();
            Activator = new ViewModelActivator();
            this.WhenActivated(disposables =>
            {
                HandleActivate();
                Disposable
                    .Create(HandleDeactivate)
                    .DisposeWith(disposables);
            });
        }

        public float AnimationValue { get; private set; }

        public int Generation { get; private set; }

        public int Run { get; private set; }

        public WinnerViewModel WinnerByFitnessFromAllRuns { get; private set; }

        public WinnerViewModel WinnerBySimplicityFromAllRuns { get; private set; }

        public WinnerViewModel WinnerByFitness { get; private set; }

        public WinnerViewModel Leader { get; private set; }

        public ObservableCollection<Measurement> FitnessGraph { get; }

        public ObservableCollection<Measurement> ComplexityGraph { get; }

        public int GenerationsCount { get; } = 3000;

        // TODO: selector ToggleButton[IsChecked=true] does not work, because bool? IsChecked cannot be converted to bool
        public bool IsPaused { get; private set; }

        public ViewModelActivator Activator { get; }

        public void TogglePause()
        {
            if (!_runSearchLoop.WaitOne(0))
            {
                _runSearchLoop.Set();
                IsPaused = false;
            }
            else
            {
                _runSearchLoop.Reset();
                IsPaused = true;
            }

            this.RaisePropertyChanged(nameof(IsPaused));
        }

        private void HandleActivate()
        {
            StartSearchLoop();
            StartPeriodicalUpdate();
        }

        private void HandleDeactivate()
        {
            CancelSearchLoop();
            CancelPeriodicalUpdate();
        }

        private void HandleStartNewSearchRun()
        {
            FitnessGraph.Clear();
            ComplexityGraph.Clear();

            ++Run;
            this.RaisePropertyChanged(nameof(Run));

            Generation = 1;
            this.RaisePropertyChanged(nameof(Generation));
        }

        private void HandleNewPopulationFound(SearchResult searchResult)
        {
            double Average(double average, double add)
            {
                return (average * _measureCount + add) / (_measureCount + 1);
            }

            var fitnessRating = searchResult.FitnessRating;
            _fitnessMeasure.Minimum = Average(_fitnessMeasure.Minimum, fitnessRating[fitnessRating.Count - 1].Fitness);
            _fitnessMeasure.Maximum = Average(_fitnessMeasure.Maximum, fitnessRating[0].Fitness);
            _fitnessMeasure.Value = Average(_fitnessMeasure.Value, fitnessRating.Average(x => x.Fitness));
            _fitnessMeasure.Time = Generation;

            var simplicityRating = searchResult.SimplicityRating;
            _complexityMeasure.Minimum = Average(_complexityMeasure.Minimum, simplicityRating[0].Complexity);
            _complexityMeasure.Maximum = Average(_complexityMeasure.Maximum,
                simplicityRating[simplicityRating.Count - 1].Complexity);
            _complexityMeasure.Value = Average(_complexityMeasure.Value, simplicityRating.Average(x => x.Complexity));
            _complexityMeasure.Time = Generation;

            if (FitnessGraph.Count == 0 || ++_measureCount > 20)
            {
                FitnessGraph.Add(_fitnessMeasure);
                ComplexityGraph.Add(_complexityMeasure);
                _measureCount = 0;
            }

            var newWinner = searchResult.FitnessRating[0];
            var newWinnerEvaluation = XorNetworkSearch.EvaluateWinner(newWinner);

            if (_leaderOfRun == null || IsBetterFitness(_leaderOfRun.ParetoFrontPoint, newWinner))
            {
                _leaderOfRun = new WinnerViewModel(
                    newWinner,
                    newWinnerEvaluation,
                    Run, Generation);
            }
            
            if (Leader == null || Leader.ParetoFrontPoint != newWinner)
            {
                Leader = new WinnerViewModel(
                    newWinner,
                    newWinnerEvaluation,
                    Run, Generation);
                this.RaisePropertyChanged(nameof(Leader));
            }
            
            if (WinnerByFitnessFromAllRuns == null ||
                IsBetterFitness(WinnerByFitnessFromAllRuns.ParetoFrontPoint, Leader.ParetoFrontPoint))
            {
                WinnerByFitnessFromAllRuns = Leader;
                this.RaisePropertyChanged(nameof(WinnerByFitnessFromAllRuns));
            }
            
            ++Generation;
            this.RaisePropertyChanged(nameof(Generation));
        }

        private void HandleFinishSearchRun()
        {
            WinnerByFitness = _leaderOfRun;
            this.RaisePropertyChanged(nameof(WinnerByFitness));
            
            if (WinnerBySimplicityFromAllRuns == null ||
                IsBetterComplexity(WinnerBySimplicityFromAllRuns.ParetoFrontPoint, WinnerByFitness.ParetoFrontPoint))
            {
                WinnerBySimplicityFromAllRuns = WinnerByFitness;
                this.RaisePropertyChanged(nameof(WinnerBySimplicityFromAllRuns));
            }

            FitnessGraph.Add(_fitnessMeasure);
            ComplexityGraph.Add(_complexityMeasure);
            _leaderOfRun = null;
            _measureCount = 0;
        }

        private static bool IsBetterComplexity(ParetoFrontPoint curr, ParetoFrontPoint candidate)
        {
            return curr.Complexity > candidate.Complexity ||
                   curr.Complexity == candidate.Complexity && curr.Fitness < candidate.Fitness;
        }

        private static bool IsBetterFitness(ParetoFrontPoint curr, ParetoFrontPoint candidate)
        {
            return curr.Fitness < candidate.Fitness ||
                   Math.Abs(curr.Fitness - candidate.Fitness) <= float.Epsilon && curr.Complexity > candidate.Complexity;
        }

        private void StartSearchLoop()
        {
            var cancellationToken = _cancellationTokenSource.Token;
            var handlesToWait = new[] {cancellationToken.WaitHandle, _runSearchLoop};

            void SearchLoop()
            {
                while (true)
                {
                    var searchModel = new XorNetworkSearch(
                        PopulationParameters,
                        NetworkParameters,
                        ReproductionParameters);

                    SearchResult searchResult = default;
                    InvokeOnUiThread(HandleStartNewSearchRun);
                    for (var generation = 0; generation < GenerationsCount; ++generation)
                    {
                        WaitHandle.WaitAny(handlesToWait);
                        if (cancellationToken.IsCancellationRequested)
                            return;

                        var newSearchResult = searchModel.SearchNext();
                        InvokeOnUiThread(() => HandleNewPopulationFound(newSearchResult));
                        searchResult = newSearchResult;
                    }

                    InvokeOnUiThread(() => HandleFinishSearchRun());
                }
            }

            _searchThread = new Thread(SearchLoop);
            _searchThread.Start();
        }

        private void CancelSearchLoop()
        {
            _cancellationTokenSource.Cancel();
            _searchThread.Join();
        }

        private void StartPeriodicalUpdate()
        {
            var lastUpdateTime = DateTime.Now.Ticks;
            
            void TimerCallback()
            {
                var currentTime = DateTime.Now.Ticks;
                AnimationValue += (currentTime - lastUpdateTime) * AnimationSpeed * PeriodicalUpdateTimeoutMilliseconds / 1000000f;
                this.RaisePropertyChanged(nameof(AnimationValue));
                lastUpdateTime = currentTime;
            }

            _periodicalUpdate = new Timer(
                state => InvokeOnUiThread(TimerCallback),
                null,
                0,
                PeriodicalUpdateTimeoutMilliseconds);
        }

        private void CancelPeriodicalUpdate()
        {
            _periodicalUpdate.Dispose();
        }

        private static void InvokeOnUiThread(Action action)
        {
            Dispatcher.UIThread.InvokeAsync(
                action,
                DispatcherPriority.Background);
        }
    }
}