function [zpt,MSE] = simulate_reservoir_network(N,g,ftoutput, ft,ht, m, mh, simtime2, simtime, simtime_len, draw)
   
    linewidth = 3;
    fontsize = 14;
    fontweight = 'bold';

    
    dt = 0.05;
    nsecs = 2000;
      
    sigma = .1;
        
    % Normalize weights

    ft2 = ft;
    x0 = 0.1 * randn(N, 1);

    %g = 3;
    J = g * randn(N, N) / sqrt(N); % Recurrent weight matrix
   
    % Simulate network
    x = x0;
    ti = 0;
    X = nan(N, simtime_len); % Preallocate a matrix to store the network states over time
    
    for t = simtime
        ti = ti + 1;
        X(:, ti) = x;
        x = (1.0 - dt) * x + J * (tanh(x) * dt) + m * (ft(ti) * dt) + mh * (ht(ti) * dt);
        %display(x);
    end

    transient = simtime_len / 2;
    X = X(:, transient + 1:end);
    ft_ss = ftoutput(transient + 1:end);
    ht_ss = ht(transient + 1:end);
    
    %% Linear Least Squares solution
    n = pinv((X + sigma * randn(size(X)))') * (ft_ss)'; % decoding weights
    %display(n);
    
    % Now test
    ti = 0;
    X2 = nan(N, simtime_len); % State of the neural network at each time step during the testing phase
    x = x0;
    zpt = nan(1, simtime_len);

    for t = simtime
        ti = ti + 1;
        x = (1.0 - dt) * x + J * (tanh(x) * dt) + m * (ft2(ti) * dt) + mh * (ht(ti) * dt);
        X2(:, ti) = x; % State of the network at this time step
        z = n' * x; % The output, decoding
        zpt(ti) = z; % Updating z
    end
    
    % Phase shift
    azpt = zpt(transient:end); % Steady states
    aft2 = ft2(transient:end); % Extracting the first half section of the stimulation time
    [xc, lags] = xcorr(azpt, aft2, 50);
    [~, b] = max(xc);
    shift = lags(b);

    if shift >= 0
        azpt = azpt(shift + 1:end);
        aft2 = aft2(1:end - shift);
    else
        aft2 = aft2(-shift + 1:end);
        azpt = azpt(1:end + shift);
    end

    error = mean(abs(azpt - aft2).^2);
    MSE = error;

    % Plot target and actual output
    if draw
        figure;
        plot(aft2(1:1000)); hold on; 
        plot(ht(1:1000))
        plot(azpt(1:1000));
        legend('Target', 'Heartbeat signal','Network output');
        title(['Testing MSE: ' num2str(error)]);
        hold off;
    end
    
end

% Example usage
