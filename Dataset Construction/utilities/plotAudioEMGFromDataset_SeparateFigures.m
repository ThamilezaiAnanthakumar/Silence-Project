function plotAudioEMGFromDataset_SeparateFigures(Dataset, wordIdx, channelList)
% plotAudioEMGFromDataset_SeparateFigures
%
% Plots Audio and EMG analyses as SEPARATE figures
% with frequency-axis limits applied.
%
% EMG freq range   : 0–5 Hz
% Audio freq range : 0–800 Hz

    if wordIdx < 1 || wordIdx > numel(Dataset)
        error('Invalid word index');
    end

    D = Dataset(wordIdx);

    %% AUDIO

    % Audio waveform
    figure('Name',['Audio Waveform - ', D.word]);
    plot(D.Audio.time, D.Audio.signal(:,1))
    xlabel('Time (s)')
    ylabel('Amplitude')
    title(['Audio Waveform - Word: ', D.word])
    grid on

    % Audio spectrogram (0–800 Hz)
    figure('Name',['Audio Spectrogram - ', D.word]);
    imagesc(D.Audio.spectrogram.T, ...
            D.Audio.spectrogram.F, ...
            abs(D.Audio.spectrogram.S))
    axis xy
    ylim([0 800])
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title(['Audio Spectrogram (0–800 Hz) - Word: ', D.word])
    colorbar

    %% EMG 

    for ch = channelList

        if ch > numel(D.EMG.channel)
            warning('EMG channel %d does not exist. Skipping.', ch);
            continue;
        end

        % EMG waveform
        figure('Name',sprintf('EMG Waveform - %s - Ch %d', D.word, ch));
        plot(D.EMG.time, D.EMG.signal(ch,:))
        xlabel('Time (s)')
        ylabel('Amplitude')
        title(sprintf('EMG Waveform - Word: %s | Channel %d', D.word, ch))
        grid on

        % EMG spectrogram (0–5 Hz)
        figure('Name',sprintf('EMG Spectrogram - %s - Ch %d', D.word, ch));
        imagesc(D.EMG.channel(ch).spectrogram.T, ...
                D.EMG.channel(ch).spectrogram.F, ...
                abs(D.EMG.channel(ch).spectrogram.S))
        axis xy
        ylim([0 5])
        xlabel('Time (s)')
        ylabel('Frequency (Hz)')
        title(sprintf('EMG Spectrogram (0–30 Hz) - Channel %d', ch))
        colorbar
     
        % EMG FFT (0–5 Hz)
        figure('Name',sprintf('EMG FFT - %s - Ch %d', D.word, ch));
        plot(D.EMG.channel(ch).FFT.frequency, ...
             D.EMG.channel(ch).FFT.magnitude)
        xlim([0 5])
        xlabel('Frequency (Hz)')
        ylabel('Magnitude')
        title(sprintf('EMG FFT (0–5 Hz) - Channel %d', ch))
        grid on

        % EMG CWT (0–5 Hz)
        figure('Name',sprintf('EMG CWT - %s - Ch %d', D.word, ch));
        imagesc(D.EMG.time, ...
                D.EMG.channel(ch).CWT.frequency, ...
                abs(D.EMG.channel(ch).CWT.coeff))
        axis xy
        ylim([0 5])
        xlabel('Time (s)')
        ylabel('Frequency (Hz)')
        title(sprintf('EMG CWT (0–5 Hz) - Channel %d', ch))
        colorbar
        colormap(jet(512))      % higher resolution colors
        caxis([2 15])            % fix color range

    end
end