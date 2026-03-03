function scrollableAudioViewer(data, Fs, windowLength)
% Scrollable audio viewer with LIVE update (slider in seconds)

    data = data(:);
    N = length(data);

    totalDuration = N / Fs;                 % total audio length (sec)
    t = (0:N-1) / Fs;

    windowSamples = round(windowLength * Fs);
    maxStartSec = totalDuration - windowLength;

    % Initial window
    startSec = 0;
    idx = 1 : windowSamples;

    % Figure
    figure('Name','Scrollable Audio Viewer','NumberTitle','off');

    hPlot = plot(t(idx), data(idx));
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['Audio Signal (' num2str(windowLength) ' s Window)']);
    grid on;
    xlim([0 windowLength]);

    % Slider (SECONDS)
    sld = uicontrol('Style','slider', ...
        'Min',0, ...
        'Max',maxStartSec, ...
        'Value',0, ...
        'Units','normalized', ...
        'Position',[0.1 0.02 0.8 0.04], ...
        'SliderStep',[1/maxStartSec 10/maxStartSec]);

    % Live update listener
    addlistener(sld,'Value','PostSet',@(~,~) updatePlot());

    function updatePlot()
        startSec = sld.Value;
        startIdx = round(startSec * Fs) + 1;
        idx = startIdx : startIdx + windowSamples - 1;

        set(hPlot,'XData',t(idx),'YData',data(idx));
        xlim([t(idx(1)) t(idx(end))]);
        drawnow limitrate
    end
end
