

function [zpt,MSE] =  reservoir_simulation_noHB(N,g, ftarget,ft, m, simtime2, simtime, simtime_len,draw )

    linewidth = 3;
    fontsize = 14;
    fontweight = 'bold';
    
    N = 100;
    nsecs = 2000;
    dt = 0.05;
 
    sigma = .1;
    
   
    x0 = 0.1*randn(N,1);
   
    J = g*randn(N,N)/sqrt(N);
    
    ft2 = ft;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Simulate network
    
    x = x0;
    ti = 0;
    X = nan(N,simtime_len);
    
    for t = simtime
        ti = ti+1;
    
        X(:,ti) = x;
        x = (1.0-dt)*x + J*(tanh(x)*dt) + m*(ft(ti)*dt);
    end
    
    transient = simtime_len/2;
    X = X(:,transient+1:end);
    ft_ss = ftarget(transient+1:end);
    
    % plot activity of first 10 neurons
    if draw
        figure; plot(X(1:10,1:1000)');
        xlabel('timepoint'); ylabel('Firing rate');
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear Least Squares solution
    n = pinv((X + sigma*randn(size(X)))')*ft_ss';
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now test.
    ti = 0;
    X2 = nan(N,simtime_len);
    % x = x0;
    zpt = nan(1,simtime_len);
    for t = simtime			
        ti = ti+1;
        x = (1.0-dt)*x + J*(tanh(x)*dt) + m*(ft2(ti)*dt);
        X2(:,ti) = x;
        z = n'*x;
        zpt(ti) = z;
    end
   
    % phase shift
    azpt = zpt(transient:end);
    aft2 = ftarget(transient:end);
    [xc, lags] = xcorr(azpt,aft2,50);
    [~, b] = max(xc);
    shift = lags(b);
    if shift >= 0
        azpt = azpt(shift+1:end);
        aft2 = aft2(1:end-shift);
    else
        aft2 = aft2(-shift+1:end);
        azpt = azpt(1:end+shift);
    end
    error = mean(abs(azpt-aft2).^2);
    MSE=error;
    %%%%%%%%%%%%%%%%%%%
    % Plot target and actual output
    if draw
        figure; plot(aft2(1:1000)); hold on;
        plot(azpt(1:1000));
        legend('Target', 'Network output')
        title(['Testing MSE: ' num2str(error)]);
    end 
 
