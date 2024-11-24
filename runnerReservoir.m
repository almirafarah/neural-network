disp('Clearing workspace.');
clear; close all; clc
tic

N = 100;
freq = pi/2;
nsecs = 2000;
dt = 0.05;

simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;


m = initializeWeights(N,1.0,0,1,0);
mh = initializeWeights(N,1.0,0,1,0);


num_trials = 1000;
steps = 25;
stepsPer = 10;
basisFrequency = floor(steps / 2) * 0.05 * pi;
fre1 = basisFrequency;
ft = cos(basisFrequency * simtime);

pulse = heartbeat_signal(simtime, basisFrequency);
figure;
plot(pulse(1:1000));hold on;
plot(ft(1:1000));

hold off;

stepsX= 25;
stepsY = 10;

MSEtoMLE(N, m, mh, simtime2, simtime, simtime_len, 1000)
MSEtowh(N, ft, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq)
MSEtoPerHBandWHfreq(N, ft,pulse,m ,mh, simtime2,simtime,simtime_len, num_trials,steps,freq ,stepsPer);
MSEtoinputPer(N, mh, simtime2, simtime, simtime_len,steps,stepsPer);
MSEtoinputStrength(N,ft,pulse, m, mh, simtime2, simtime, simtime_len,  num_trials,freq,steps);
MSEtoHBPerandWfreq(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq,stepsPer );
MSEtofunctionw(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq );
MSEtoChaosMultiHB(N,ft,pulse, m, mh, simtime2,simtime,simtime_len, num_trials,stepsX,stepsY,freq );

toc



function MSEtoMLE(N, m, mh, simtime2, simtime, simtime_len,num_trials)
    for i= 1:3 % wh
        for j = 1:3 %w
            for z = 1:3 %g
                g = 0.5 +z - 1;
                freqw = 0.05 + (j-1) * (0.66 - 0.05);
                ft = cos(freqw *simtime);
                ft2 = cos(freqw *simtime);
                freqwh = 0.05 +(i-1) * (0.66 - 0.05);
                pulse = heartbeat_signal(simtime,freqwh);
                [outputfunction, errorinstance(i)] = simulate_reservoir_network(N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, true);
                errors_with_heartbeat = zeros(1,num_trials);
                parfor trial = 1:num_trials
                    [final_states_with_heartbeat, errors_with_heartbeat(trial)] = simulate_reservoir_network(N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false);
                end
                disp(['Mean MSE with heartbeat: ' num2str(mean(errors_with_heartbeat)) ,'g is ', num2str(g),'input frequency w is ',num2str(freqw), 'heartbeat freq wh is ', num2str(freqwh)]);
                
            end
        end
    end
end 

function heartbeat = heartbeat_signal(t, freq)
    period = 2 * pi / freq;  % period of the heartbeat signal
    heartbeat = abs(mod(t, period) - period/2) / (period/2);
end

function heartbeat = heartbeat_signal2(t, freq)
    fs = 1e12; % Sampling frequency, set very high (1 THz) for accurate representation
    fc = 20e9; % Center frequency, set to 10 GHz
    bw = 1; % Fractional bandwidth set to a high value to get a narrow pulse
    
    % Calculate the cutoff time to achieve 1 ns pulse width
    tc = gauspuls('cutoff', fc, bw, [], -40); % Cutoff time for the pulse
    
    % Time vector spanning approximately 1 ns
    t2 = -tc:1/fs:tc; % Time vector with 1 ps (picosecond) steps
    
    % Find the indices where cos(2*pi*fc*t) > 0 and cos(2*pi*fc*t) < 0
    cos_t = cos(2*pi*fc*t2);
    idx_pos = find(cos_t > 0);
    idx_neg = find(cos_t < 0);
    
    % Adjust the time vector t to span exactly the time from when cos > 0 to when cos < 0
    t_adj = t2(idx_pos(1):idx_neg(end));
    
    % Generate the Gaussian-modulated pulse x using the adjusted time vector
    x = gauspuls(4*t_adj, fc, bw);

        
    % Calculate the phase shift to align the heartbeat signal with the cosine function
    phase_shift = idx_pos(1) - 1;

    period = 2 * pi / freq;  % period of the heartbeat signal
    heartbeat = zeros(size(t));

    for i = 1:length(t)
        heartbeat(i) = x(mod(i-1 + phase_shift, length(x))+1);
    end

end

function errortest(N,inputft,ft,pulse, m, mh, simtime2, simtime, simtime_len,  num_trials,freq)
        % Run simulations with heartbeat
    for i = 1:num_trials
        [final_states_with_heartbeat, error] = simulate_reservoir_network(N,cos(0.5 * freq * simtime),ft,pulse, m, mh, simtime2, simtime, simtime_len, false);
        [final_states_with_heartbeat, errors_with_heartbeat(i)] = simulate_reservoir_network(N,cos(0.5 * freq * simtime),final_states_with_heartbeat,pulse, m, mh, simtime2, simtime, simtime_len,false);
    end
    
    % Run simulations without heartbeat
    for i = 1:num_trials
        [final_states_without_heartbeat,error] = reservoir_simulation_noHB(N,cos(0.5 * freq * simtime),cos(0.5 * freq * simtime), m, simtime2, simtime, simtime_len,false );
        [final_states_without_heartbeat,errors_without_heartbeat(i)] = reservoir_simulation_noHB(N,cos(0.5 * freq * simtime), final_states_without_heartbeat, m, simtime2, simtime, simtime_len, false );
    end
    % Perform paired t-test
    [h, p] = ttest(errors_with_heartbeat, errors_without_heartbeat);
    
    % Display the results
    disp(['Mean MSE with heartbeat: ' num2str(mean(errors_with_heartbeat))]);
    disp(['Mean MSE without heartbeat: ' num2str(mean(errors_without_heartbeat))]);
    disp(['p-value from paired t-test: ' num2str(p)]);
    
    if h == 1
        disp('There is a significant difference in MSE with and without heartbeat.');
    else
        disp('There is no significant difference in MSE with and without heartbeat.');
    end
        % Perform paired t-test
    [h, p] = ttest(errors_with_heartbeat, errors_without_heartbeat);
    
    % Display the results
    disp(['Mean MSE with heartbeat: ' num2str(mean(errors_with_heartbeat))]);
    disp(['Mean MSE without heartbeat: ' num2str(mean(errors_without_heartbeat))]);
    disp(['p-value from paired t-test: ' num2str(p)]);
    
    if h == 1
        disp('There is a significant difference in MSE with and without heartbeat.');
    else
        disp('There is no significant difference in MSE with and without heartbeat.');
    end

end 

function inputweights =  initializeWeights(N,range, percentage_excitatory, comp_random, selectNeurons, g)


    if comp_random == 1
        m = randn(N, 1);
        m = m/norm(m);
        inputweights = m;
    end
    
    if selectNeurons > 0 %when select neurons is negative then the weights are selected randomly to all the neurons 
     
        random_vector = randn(N, 1);  % Example: Normally distributed random number
        num_zeros = round(selectNeurons * N);
        zero_indices = randperm(N, num_zeros);
        random_vector(zero_indices) = 0;
        m = random_vector/norm(random_vector);

    end 
    if comp_random == 0
          
        % Calculate number of excitatory and inhibitory neurons
        num_excitatory = round(N * percentage_excitatory);
        num_inhibitory = N - num_excitatory;
        
        % Ensure the percentages add up to 100%
        if num_excitatory + num_inhibitory ~= N
            error('The sum of excitatory and inhibitory neurons must equal the total number of neurons.');
        end
        
        % Initialize the weight vector
        inputweights = zeros(N, 1);
        
        % Assign excitatory weights (positive values)
        excitatory_weights = randn(num_excitatory, 1)*range;  % example range scaled by standard deviation
        
        % Normalize excitatory weights
        excitatory_weights = excitatory_weights / norm(excitatory_weights);
        
        % Assign inhibitory weights (negative values)8rsa
        inhibitory_weights = -randn(num_inhibitory, 1)*range;  % example range scaled by standard deviation
        
        % Normalize inhibitory weights
        inhibitory_weights = inhibitory_weights / norm(inhibitory_weights);
        
        % Combine excitatory and inhibitory weights
        inputweights(1:num_excitatory) = excitatory_weights;
        inputweights(num_excitatory+1:end) = inhibitory_weights;
        
        % Shuffle the order of neurons to avoid bias
        perm = randperm(N);
        inputweights = inputweights(perm);
        
        % Display the input weights vector (optional)
        %disp('Input Weights:');
        %disp(inputweights);
    
        
        
    end
end 

function MSEtoMEAN(N,inputft,ft, m, mh, simtime2, simtime, simtime_len, num_trials,freq)
    XwithHB = zeros(1,8);
    XwithoutHB = zeros(1,8);
    connectivity = 1;
    g = 0.5;

    while connectivity<=8
        m = initializeWeights(N,1,0,1,g);
        

        for i = 1:num_trials
            [final_states_with_heartbeat, error] = simulate_reservoir_network(N,cos(1.0 * freq * simtime),ft, m, mh, simtime2, simtime, simtime_len, false);
            [final_states_with_heartbeat, errors_with_heartbeat(i)] = simulate_reservoir_network(N,cos(1.0 * freq * simtime),final_states_with_heartbeat, m, mh, simtime2, simtime, simtime_len,false);
        end
        display(num2str(errors_with_heartbeat));
        XwithHB(:,connectivity)= mean(errors_with_heartbeat);
        % Run simulations without heartbeat
        for i = 1:num_trials
            [final_states_without_heartbeat,error] = reservoir_simulation_noHB(N,cos(1.0 * freq * simtime),cos(1.0 * freq * simtime), m, simtime2, simtime, simtime_len,false );
            [final_states_without_heartbeat,errors_without_heartbeat(i)] = reservoir_simulation_noHB(N,cos(1.0 * freq * simtime), final_states_without_heartbeat, m, simtime2, simtime, simtime_len, false );
        end
        XwithoutHB(:,connectivity)= mean(errors_without_heartbeat);
        
        connectivity= connectivity+1;
        g = g+0.1;
    end 
    figure;
    plot(XwithHB(1:8)); hold on;
    plot(XwithoutHB(1:8));
    legend('ْwith heartbeat', 'without heartbeat')
    %title(['Testing MSE: ' num2str(error)]);
    


end 

function graph = MSEtoinputStrength(N,ft,pulse, m, mh, simtime2, simtime, simtime_len,  num_trials,freq,steps)
    figure;hold on;
    e = zeros(1,steps);
    error = zeros(1,num_trials);
    g= 1.2;
    ft2 = ft;
    parfor i = 1:25
       
        e(1,i) = trials(error,N,g,ft2,ft,pulse,(i+i*0.05)*m(1:length(m)), mh, simtime2, simtime, simtime_len, false)
        
    end
    graph = e;
    plot(e(1,:)); hold on;
    set(gca, 'XTick' ,1:steps,'XTickLabel',0.05: 0.05: 0.05*steps);
    xlabel('input function frequency "w"');
    ylabel('MSE');
    title('MSE as a function of input fucntion frequency');
    legend('show'); % Show the legend to identify each row
    hold off;   

end

% generate heartbeat according to input frequency. fit hertbeat function to
% input fuction
function pulse = simulateHeartBeat(amp,freq,draw)
    
    simtime = 0:dt:nsecs-dt;
    simtime_len = length(simtime);
    simtime2 = 1*nsecs:dt:2*nsecs-dt;
    
    ft = amp * cos(1*freq*simtime);
    %display(find(ft==1));
    fs = 1e12; % Sampling frequency, set very high (1 THz) for accurate representation
    fc = 20e9; % Center frequency, set to 10 GHz
    bw = 1; % Fractional bandwidth set to a high value to get a narrow pulse
    
    % Calculate the cutoff time to achieve 1 ns pulse width
    tc = gauspuls('cutoff', fc, bw, [], -40); % Cutoff time for the pulse
    
    % Time vector spanning approximately 1 ns
    t = -tc:1/fs:tc; % Time vector with 1 ps (picosecond) steps
    
    % Generate the Gaussian-modulated pulse
    x = gauspuls(4*t, fc, bw);
    
    %%% take a range to put the heartbeat signal and all everything outside the
    %%% range make zero. you can make the heartbeat function exactly like the
    %%% cos function and then make it pulses of 1. or dress x on the required
    %%% range. 
    listofOnes = find(ft==1);
    pulse = zeros(1,simtime_len);
    middle = length(x)/2;
    for i = 1:length(listofOnes)
        %if mod(i, 2) ~= 0 
         %   continue
        %end
        j = listofOnes(i);
        from = min(middle - 1, j - 1);
        to = min(middle, length(pulse) - j);
        pulse(j - from : j + to) = x(middle - from : middle + to);
    end
    
      
    old = false;
    if old
        fs = 1e12; % Sampling frequency, set very high (1 THz) for accurate representation
        fc = 10e9; % Center frequency, set to 10 GHz
        bw = 1; % Fractional bandwidth set to a high value to get a narrow pulse
        
        % Calculate the cutoff time to achieve 1 ns pulse width
        tc = gauspuls('cutoff', fc, bw, [], -40); % Cutoff time for the pulse
        
        % Time vector spanning approximately 1 ns
        t = -tc:1/fs:tc; % Time vector with 1 ps (picosecond) steps
        
        % Generate the Gaussian-modulated pulse
        x = amp* gauspuls(4*t, fc, bw);
        pulse = x;
        for i = 1:175
            if i == 175
                pulse = [pulse,x(1:100)];
            else
                pulse = [pulse,x];
            end 
                 
        end
        Ht= pulse;
        draw = true;
        if draw == true
            figure;
            plot(pulse);
            
            
            %Plot the res5ult to visualize the pulse
            figure;
            plot(t, x);
            title('Gaussian Pulse with 1 ns Width');
            xlabel('Time (seconds)');
            ylabel('Amplitude');
            grid on;
        end
    end

end

function MSEtoinputConnectivityEXcitation(N,ft, simtime2, simtime, simtime_len,  num_trials,freq)
    
    for excRatio = 1:10
        fPlot(excRatio,:) = MSEtoStrength(N,ft, initializeWeights(N,1.0,excRatio/10,false,1.0), initializeWeights(N,1.0,excRatio/10,false,1.0) , simtime2, simtime, simtime_len,  num_trials,freq);   
    end
    figure;
    hold on; % Hold on to plot multiple rows on the same graph  
    for i = 1:size(fPlot, 1)
        plot(fPlot(i, :), 'DisplayName', ['Excititory Connection percentage ' num2str(i/10)]);
    end
    
    xlabel('Input connectivity strength');
    ylabel('Excititory neurons percentage');
    title('MSE as a function of input strength for every excitirory neurons ');
    legend('show'); % Show the legend to identify each row
    hold off;
        

end

function MSEtoConnectivityEXcitation(N,ft, simtime2, simtime, simtime_len,  num_trials,freq)
    
    for excRatio = 1:10
        fPlot(excRatio,:) = MSEtoinputStrength(N,ft, initializeWeights(N,1.0,excRatio/10,false,1.0), initializeWeights(N,1.0,excRatio/10,false,1.0) , simtime2, simtime, simtime_len,  num_trials,freq);
   
    end
    figure;
    hold on; % Hold on to plot multiple rows on the same graph
    
    for i = 1:size(fPlot, 1)
        plot(fPlot(i, :), 'DisplayName', ['Excititory Connection percentage ' num2str(i/10)]);
    end
    
    xlabel('Input connectivity strength');
    ylabel('Excititory neurons percentage');
    title('MSE as a function of input strength for every excitirory neurons ');
    legend('show'); % Show the legend to identify each row
    hold off;        
end

function graph = MSEtoStrength(N,ft, m, mh, simtime2, simtime, simtime_len,  num_trials,freq)
    for i = 1:num_trials
        for j = 1:5
            [f, error(j)] = simulate_reservoir_network(N,cos(1.0 * freq * simtime),ft,simulateHeartBeat(1), (i+i*0.05)*m(1:length(m)),(i+i*0.05)* mh(1:length(mh)), simtime2, simtime, simtime_len, false);
            e(i)= mean(error);
        end 
    end
    graph = e;
end

function MSEtoChaoswithHB(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,stepsX,freq )
    g = 0;
    for j = 1:stepsX
        g = g + 0.1;
        for i=1:num_trials
             [outputfunction, errorinstance(i)] = simulate_reservoir_network(N,g,ft,ft,pulse, m, mh, simtime2, simtime, simtime_len, false);
        end
        error(j) = mean(errorinstance);
    end
    figure;
    plot(error);
    xlabel('chaos factor "g"');
    ylabel('MSE');
    title('MSE as a function of inner reservoir connectivity strength ');
    %legend('show'); % Show the legend to identify each row
    hold off;
end 

function plotfunction(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,stepsX,stepsY,freq )
    for j = 1:floor(stepsX/4) % change wh freq
            freq = 0.05*pi*j*4;
            pulse = heartbeat_signal(simtime, freq);
            [outputfunction, error] = simulate_reservoir_network(N,1,ft,ft,pulse, m, mh, simtime2, simtime, simtime_len, true);
    end
end

%MSE as a function of chaos "g", when HB doesn't match the input frequency.
function MSEtoChaosMultiHB(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,stepsX,stepsY,freq )
    figure;hold on;
    ft2 = ft;
    error = zeros(floor(stepsX/4),stepsX);
    g=0;
    for z = 1:floor(stepsX/4) % change wh freq
        freq = 0.05*pi*z*4;    
        pulse = heartbeat_signal(simtime, freq);
        errorg(z,:)= stepsGchnage(N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false,25);
        plot(errorg(z, :), 'DisplayName', [' w HB = 0.05 * pi * ' num2str(j*4)]); 
        hold on;
    end 
    set(gca, 'XTick' ,1:stepsX,'XTickLabel',0:0.1:stepsX*0.1);
    xlabel('chaos factor "g"');
    ylabel('MSE');
    title('MSE as a function of inner reservoir connectivity strength ');
    legend('show'); % Show the legend to identify each row
    hold off;
end

function MSEtosingleWHBnormalG(N, ft, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq)
figure; hold on;
    ft2 = ft;
    g = 0;
    error=zeros(1,steps+15);
    for z = 1:1
        freq = 0;
        g = 0.5 * z;
        errorz = stepsfreqChange(error,z,N,g,ft2,ft, m, mh, simtime2, simtime, simtime_len,(steps+15),true,false);
        disp(errorz(1,:));
        error(z,:) = errorz(1,:);
        %disp("yes");
        %disp(error(z,:));
    end
    for z = 1:1 
        plot(error(z,:), 'DisplayName', [' g =' num2str((0.5*4))]); hold on;
    end
    set(gca, 'XTick' ,1:(steps+15),'XTickLabel',0.05: 0.05 : 0.05*(steps+15));
    xlabel('HeartBeatfrequency "wh"');
    ylabel('MSE');
    title('MSE as a function of HeartBeat frequency"wh" ');
    legend('show'); % Show the legend to identify each row
    hold off;
end


function MSEtofunctionwWoHB(N, m , simtime2,simtime,simtime_len, num_trials,steps,freq )
    g = 0;
    figure;hold on;
    %colors = autumn(8);
    error = zeros(5+3, steps);
    for z = 1:5+3
        freq = 0;
        g = z * 0.5;
        error(z,:) = graphWOHB(N,g, m, simtime2, simtime, simtime_len,false,steps, num_trials);
        plot(error(z, :), 'DisplayName', [' g =' num2str(g)]); hold on;
    end
    set(gca, 'XTick' ,1:steps,'XTickLabel',0.05: 0.05: 0.05*(steps));
    xlabel('input function frequency "w"');
    ylabel('MSE');
    title('MSE as a function of input fucntion frequency');
    legend('show'); % Show the legend to identify each row
    hold off;
end

function meantrial = trialWOHB(N,g, ft2,ft, m, simtime2, simtime, simtime_len,false,num_trials) 
    parfor i=1:num_trials
        [outputfunction, errorinstance(i)] = reservoir_simulation_noHB(N,g, ft2,ft, m, simtime2, simtime, simtime_len,false);
    end
    meantrial = mean(errorinstance);

end

function graphArray = graphWOHB (N,g, m, simtime2, simtime, simtime_len,false,steps,num_trials)
    graphArray = zeros(1, steps);
    freq = 0;
    parfor j = 1:steps
        freq = 0.05 * pi *j;
        ft = cos(1 * freq * simtime);
        ft2 = cos(1* freq * simtime);
        graphArray(1,j) = trialWOHB(N,g, ft2,ft, m, simtime2, simtime, simtime_len,false,num_trials);
    end
end

%chnage function frequency and see if the change in MSE
function MSEtofunctionw(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq )
    g = 0;
    figure;hold on;
    error=zeros(5,steps);
    errorz = zeros(1,steps);
    for z = 1:5
        freq = 0;
        g = 0.5 * z;
        ft2 = cos(1.0 *simtime);
        ft = cos(1.0 *simtime);
        errorz = stepsfreqChange(error,z,N,g,ft2,ft, m, mh, simtime2, simtime, simtime_len,steps,false,true);
        disp(errorz(1,:));
        error(z,:) = errorz(1,:);
    end
    for z = 1:5 
        plot(error(z,:), 'DisplayName', [' g =' num2str(g)]); hold on;
    end
    set(gca, 'XTick' ,1:steps,'XTickLabel',0.05: 0.05: 0.05*steps);
    xlabel('input function frequency "w"');
    ylabel('MSE');
    title('MSE as a function of input fucntion frequency');
    legend('show'); % Show the legend to identify each row
    hold off;
end

%change number of neurons that recieve the input function in the reservoir
%and chnage freq of the input signal
function MSEtoinputPer(N, mh, simtime2, simtime, simtime_len,steps,stepsPer)
    error = zeros(stepsPer,steps);
    g=1.2;
    for z = 1:stepsPer
        freq = 0;
        ft2 = cos(1.0 *simtime);
        ft = cos(1.0 *simtime);
        errorz = stepsfreqChange(error,z,N,g,ft2,ft, initializeWeights(N,1, 1, 0, 0.01*z, g), mh, simtime2, simtime, simtime_len,steps,false,true);
        disp(errorz(1,:));
        error(z,:) = errorz(1,:);

    end
    for z = 1:stepsPer
        plot(error(z,:), 'DisplayName', [' % neurons give input =' num2str((0.01*z))]); hold on;
    end
    set(gca, 'XTick' ,1:steps,'XTickLabel',0.05: 0.05: 0.05*steps);
    xlabel('input function frequency "w"');
    ylabel('MSE');
    title('MSE as a function of input fucntion frequency');
    legend('show'); % Show the legend to identify each row
    hold off;    
end

%which percent of the neurons recieve the heartbeat signal not the input, while
%increasing input frequncy. 
function MSEtoHBPerandWfreq(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq ,stepsPer)
    error = zeros(stepsPer,steps);
    g=1.2;
    for z = 1:stepsPer
        freq = 0;
        ft2 = cos(1.0 *simtime);
        ft = cos(1.0 *simtime);
        errorz = stepsfreqChange(error,z,N,g,ft2,ft, m, initializeWeights(N,1, 1, 0, 0.01*z, g), simtime2, simtime, simtime_len,steps,false,true);
        disp(errorz(1,:));
        error(z,:) = errorz(1,:);
    end
    for z = 1:stepsPer
        plot(error(z,:), 'DisplayName', [' % neurons recieve HB =' num2str((0.01*z))]); hold on;
    end
    set(gca, 'XTick' ,1:steps,'XTickLabel',0.05: 0.05: 0.05*steps);
    xlabel('input function frequency "w"');
    ylabel('MSE');
    title('MSE as a function of input fucntion frequency');
    legend('show'); % Show the legend to identify each row
    hold off;   
end


%MSE to heartbeat frequency while controlling for the percentage of neurons
%taht recieve the heartbeat signal.
function MSEtoPerHBandWHfreq(N, ft,pulse,m ,mh, simtime2,simtime,simtime_len, num_trials,steps,freq ,stepsPer)
    error = zeros(stepsPer,steps);
    g=1.2;
    for z = 1:stepsPer
        freq = 0;
       ft2 = ft;
        errorz = stepsfreqChange(error,z,N,g,ft2,ft, m, initializeWeights(N,1, 1, 0, 0.01*z, g), simtime2, simtime, simtime_len,steps,true,false);
        disp(errorz(1,:));
        error(z,:) = errorz(1,:);
    end
    for z = 1:stepsPer
        plot(error(z,:), 'DisplayName', [' % neurons recieve HB =' num2str((0.01*z))]); hold on;
    end
    set(gca, 'XTick' ,1:steps,'XTickLabel',0.05: 0.05: 0.05*steps);
    xlabel('input function frequency "wh"');
    ylabel('MSE');
    title('MSE as a function of input fucntion frequency');
    legend('show'); % Show the legend to identify each row
    hold off;   
end

%change frequency of heartbeat and see if there's change in MSE
function MSEtowh(N, ft, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq)
    figure; hold on;
    ft2 = ft;
    g = 0;
    error= zeros(6,steps);
    errorz = zeros(1,steps);
    for z = 1:5
        freq = 0;
        g = 0.5 * z;
        errorz = stepsfreqChange(error,z,N,g,ft2,ft, m, mh, simtime2, simtime, simtime_len,steps,true,false);
        disp(errorz(1,:));
        error(z,:) = errorz(1,:);
        %disp("yes");
        %disp(error(z,:));
    end
    %%%check what happens at g = 1.2
        freq = 0;
        g =1.2;
        errorz = stepsfreqChange(error,z,N,g,ft2,ft, m, mh, simtime2, simtime, simtime_len,steps,true,false);
        disp(errorz(1,:));
        error(6,:) = errorz(1,:);
        %disp("yes");
        %disp(error(z,:));
        plot(error(6,:), 'DisplayName', [' g =' num2str(1.2)]); hold on;
    for z = 1:5 
        plot(error(z,:), 'DisplayName', [' g =' num2str(z*0.5)]); hold on;
    end
    set(gca, 'XTick' ,1:steps,'XTickLabel',0.05: 0.05 : 0.05*steps);
    xlabel('HeartBeatfrequency "wh"');
    ylabel('MSE');
    title('MSE as a function of HeartBeat frequency"wh" ');
    legend('show'); % Show the legend to identify each row
    hold off;
end

function error2 = trials(error,N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false)
    errorinstance = zeros(1,1000);
    parfor i=1:1000
        [outputfunction, errorinstance(i)] = simulate_reservoir_network(N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false);
    end
    error2 = mean(errorinstance);
end

function errorz = stepsfreqChange(error,z,N,g,ft2,ft, m, mh, simtime2, simtime, simtime_len, steps, changewh, changew)
    if changewh %up wh frequency 
        parfor j = 1:steps
            freq = 0.05 * pi*j;
            %pulse = heartbeat_signal(simtime, freq);
            errorz(1,j) = trials(error,N,g,ft2,ft,heartbeat_signal(simtime, freq), m, mh, simtime2, simtime, simtime_len, false);
            %disp(errorz(1,j));
        end
    end


    if changew %up w freq
        pulse =  heartbeat_signal(simtime, floor(steps / 2) * 0.05 * pi);
        errorz = zeros(1,steps);
        parfor q = 1:steps
            freq = 0.05 * pi*q;
            ft = cos(freq * simtime);
            ft2 = cos(freq * simtime);
            errorz(1,q) = trials(error,N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false);
            %disp(errorz(1,j));
        end
    end

end

function meanresult = meanresult(N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false)
    errorinstance = zeros(1,1000);
    pulse = heartbeat_signal(simtime,floor(25/4));
    parfor i=1:1000
        [outputfunction, errorinstance(i)] = simulate_reservoir_network(N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false);
    end
    meanresult = mean(errorinstance);
end

function errorg = stepsGchnage(N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false,steps)
    g= 0;
    errorg = zeros(1,steps);
    parfor j = 1:steps %change g % draw line of MSE as a function of g for this specific wh 
        g = j * 0.1;
        errorg(1,j) = meanresult (N,g,ft2,ft,pulse, m, mh, simtime2, simtime, simtime_len, false );
    end

end

%errortest(N,ft,ft,pulse, m, mh, simtime2, simtime, simtime_len,num_trials,freq);
%disp(num2str(mean(initializeWeights(N,1,0,1,1))));
%disp(num2str(mean(initializeWeights(N,1,0,1,1))));
%MSEtoMEAN(N,ft,ft, m, mh, simtime2, simtime, simtime_len,10,freq);
%MSEtoConnectivityEXcitation(N,ft, simtime2, simtime, simtime_len,  num_trials,freq)
%[f, e] = simulate_reservoir_network(N,1.1,cos(0.5 * freq * simtime),ft,pulse, m, mh, simtime2, simtime, simtime_len, true);

%reservoir_simulation_noHB(N,cos(0.5 * freq * simtime),cos(0.5 * freq * simtime), m, simtime2, simtime, simtime_len,true );

%MSEtoChaoswithHB(N, ft,pulse, m , mh, simtime2,simtime,simtime_len, num_trials,steps,freq );

%MSEtoAmp(N,ft, m, mh,pulse, simtime2, simtime, simtime_len,num_trials,freq)
%MSEtoinputStrength(N,ft, m, mh, simtime2, simtime, simtime_len,  num_trials,freq)

