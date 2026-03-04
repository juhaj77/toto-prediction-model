%% TOTO-ENNUSTE-SCRAPER (Versio 2026 - 39 COLUMNS SYNC)
clearvars; clc;

% --- SETTINGS ---
targetCardId = '443077258'; 
haluttuLahto = 11;               
options = weboptions('ContentType', 'text', 'UserAgent', 'Mozilla/5.0', 'Timeout', 20, 'CharacterEncoding', 'UTF-8');

puhdista = @(str) regexprep(strrep(strrep(strrep(strrep(strrep(strrep(str, ...
    char(228), 'a'), char(246), 'o'), char(196), 'A'), char(214), 'O'), ...
    char(229), 'a'), char(197), 'A'), ... 
    '[^a-zA-Z0-9\s\-\.\:]', ''); 

KaikkiData = {};

try
    rRaw = webread(sprintf('https://www.veikkaus.fi/api/toto-info/v1/card/%s/races', targetCardId), options);
    allRaceIds = regexp(rRaw, '"raceId":(\d+)', 'tokens');
    allRaceNums = regexp(rRaw, '"number":(\d+)', 'tokens');
    
    vId = '';
    for k = 1:min(length(allRaceIds), length(allRaceNums))
        if strcmp(allRaceNums{k}{1}, num2str(haluttuLahto)), vId = allRaceIds{k}{1}; break; end
    end
    if isempty(vId), error('Lahtoa ei loytynyt!'); end

    raceSpec = regexp(rRaw, sprintf('\\{"raceId":%s,.*?"distance":(\\d+),.*?"breed":"(.*?)".*?"startType":"(.*?)"', vId), 'tokens', 'once');
    currDist = str2double(raceSpec{1});
    isSH = double(strcmp(raceSpec{2}, 'K')); 
    currentIsAuto = double(strcmp(raceSpec{3}, 'CAR_START')); 
    
    cardInfo = webread(sprintf('https://www.veikkaus.fi/api/toto-info/v1/card/%s', targetCardId), options);
    currentDate = puhdista(char(regexp(cardInfo, '"meetDate":"(.*?)"', 'tokens', 'once')));

    hRaw = webread(sprintf('https://www.veikkaus.fi/api/toto-info/v1/race/%s/runners', vId), options);
    hBlocks = strsplit(hRaw, '{"runnerId"'); hBlocks(1) = []; 

    for i = 1:length(hBlocks)
        block = hBlocks{i};
        
        % --- ERISTETÄÄN NYKYHETKI HISTORIASTA ---
        palkit = strsplit(block, '{"priorStartId"');
        nykyhetkiBlock = palkit{1};

        nro = str2double(char(regexp(block, '"startNumber":(\d+)', 'tokens', 'once')));
        nimi = puhdista(char(regexp(block, '"horseName":"(.*?)"', 'tokens', 'once')));
        vNimi = puhdista(char(regexp(block, '"coachName":"(.*?)"', 'tokens', 'once')));
        if isempty(vNimi), vNimi = 'Unknown'; end
        
        ika = str2double(char(regexp(block, '"horseAge":(\d+)', 'tokens', 'once')));
        spRaw = char(regexp(block, '"gender":"(.*?)"', 'tokens', 'once'));
        spNum = 2; if strcmp(spRaw, 'TAMMA'), spNum = 1; elseif strcmp(spRaw, 'ORI'), spNum = 3; end
        
        currOhjNimi = puhdista(char(regexp(block, '"driverName":"(.*?)"', 'tokens', 'once')));
        if isempty(currOhjNimi), currOhjNimi = 'Unknown'; end

        % --- NYKYHETKI FEATURET ---
        k_etu = char(regexp(nykyhetkiBlock, '"frontShoes":"(.*?)"', 'tokens', 'once'));
        if isempty(k_etu), k_etu = 'UNKNOWN'; end

        k_taka = char(regexp(nykyhetkiBlock, '"rearShoes":"(.*?)"', 'tokens', 'once'));
        if isempty(k_taka), k_taka = 'UNKNOWN'; end

        k_etu_ch = double(~isempty(strfind(nykyhetkiBlock, '"frontShoesChanged":true')));
        k_taka_ch = double(~isempty(strfind(nykyhetkiBlock, '"rearShoesChanged":true')));
        
        curr_spec_cart = char(regexp(nykyhetkiBlock, '"specialCart":"(.*?)"', 'tokens', 'once'));
        if isempty(curr_spec_cart), curr_spec_cart = 'UNKNOWN'; end

        is_scratched = double(~isempty(strfind(nykyhetkiBlock, '"scratched":true')));

        % --- ENNÄTYS + Is_Auto_Record ---
        % KORJAUS: isAutoRec lasketaan ennätystyypin perusteella,
        % ei kovakoodattuna nollana. Sama logiikka kuin opetusscraperisssa.
        eMatch = regexp(block, '"(handicapRaceRecord|mobileStartRecord|vaultStartRecord)":"([\d,]+)[a-z]*"', 'tokens', 'once');
        ennatys = 0; isAutoRec = 0;
        if ~isempty(eMatch)
            ennatys = str2double(strrep(eMatch{2}, ',', '.'));
            isAutoRec = double(~isempty(strfind(eMatch{1}, 'mobile')));
        end

        peliP = 0; pMatch = regexp(block, '"percentage":(\d+)', 'tokens', 'once');
        if ~isempty(pMatch), peliP = str2double(pMatch{1}) / 100; end
        
        voittoP = 0;
        statBlock = regexp(block, '"total":\{"year":"total",.*?"starts":(\d+),.*?"position1":(\d+)', 'tokens', 'once');
        if ~isempty(statBlock)
            sC = str2double(statBlock{1}); wC = str2double(statBlock{2});
            if sC > 0, voittoP = round((wC / sC) * 100, 2); end
        end

        starts = strsplit(block, '{"priorStartId"'); starts(1) = []; 
        if isempty(starts)
            % 39 saraketta - isAutoRec laskettu, ei kovakoodattu 0
            KaikkiData(end+1, 1:39) = {vId, nro, nimi, vNimi, currOhjNimi, ika, spNum, isSH, k_etu, k_taka, ...
                                     k_etu_ch, k_taka_ch, curr_spec_cart, is_scratched, currDist, currentIsAuto, currentDate, peliP, voittoP, ennatys, isAutoRec, ...
                                     '', '', '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '', 10, 0, 0, 0};
        else
            for s = 1:length(starts)
                sB = starts{s};
                oNimi = puhdista(char(regexp(sB, '"driverFullName":"(.*?)"', 'tokens', 'once')));
                rNimi = puhdista(char(regexp(sB, '"trackCode":"(.*?)"', 'tokens', 'once')));
                
                tCond = puhdista(char(regexp(sB, '"trackCondition":"(.*?)"', 'tokens', 'once')));
                if isempty(tCond), tCond = 'unknown'; end

                kmRaw = char(regexp(sB, '"kmTime":"(.*?)"', 'tokens', 'once'));
                histIsAuto = double(~isempty(strfind(kmRaw, 'a'))); 
                isLaukka = double(~isempty(strfind(kmRaw, 'x')));
                kmNum = str2double(strrep(regexp(kmRaw, '[\d,]+', 'match', 'once'), ',', '.'));
                if isnan(kmNum), kmNum = 0; end

                h_k_etu = char(regexp(sB, '"frontShoes":"(.*?)"', 'tokens', 'once'));
                if isempty(h_k_etu), h_k_etu = 'UNKNOWN'; end

                h_k_taka = char(regexp(sB, '"rearShoes":"(.*?)"', 'tokens', 'once'));
                if isempty(h_k_taka), h_k_taka = 'UNKNOWN'; end
                
                h_spec_cart = char(regexp(sB, '"specialCart":"(.*?)"', 'tokens', 'once'));
                if isempty(h_spec_cart), h_spec_cart = 'UNKNOWN'; end

                sijRaw = lower(char(regexp(sB, '"result":"(.*?)"', 'tokens', 'once')));
                isHyl = double(~isempty(regexp(sijRaw, '[hdp]', 'once'))); 
                isKesk = double(~isempty(strfind(sijRaw, 'k')));
                sNumM = regexp(sijRaw, '^\d+', 'match', 'once');
                if ~isempty(sNumM), sijNum = str2double(sNumM);
                else if isHyl, sijNum = 20; elseif isKesk, sijNum = 21; else sijNum = 10; end
                end

                % 39 saraketta - isAutoRec laskettu, ei kovakoodattu 0
                KaikkiData(end+1, 1:39) = {vId, nro, nimi, vNimi, currOhjNimi, ika, spNum, isSH, k_etu, k_taka, ...
                                          k_etu_ch, k_taka_ch, curr_spec_cart, is_scratched, currDist, currentIsAuto, currentDate, peliP, voittoP, ennatys, isAutoRec, ...
                                          puhdista(char(regexp(sB, '"shortMeetDate":"(.*?)"', 'tokens', 'once'))), ...
                                          oNimi, rNimi, ...
                                          str2double(char(regexp(sB, '"distance":(\d+)', 'tokens', 'once'))), ...
                                          str2double(char(regexp(sB, '"startTrack":(\d+)', 'tokens', 'once'))), ...
                                          kmNum, h_k_etu, h_k_taka, h_spec_cart, histIsAuto, isLaukka, isHyl, isKesk, tCond, sijNum, ...
                                          str2double(char(regexp(sB, '"winOdd":"(\d+)"', 'tokens', 'once')))/10, ...
                                          str2double(char(regexp(sB, '"firstPrize":(\d+)', 'tokens', 'once')))/100, ...
                                          0};
            end
        end
    end 
catch ME
    fprintf('Virhe: %s\n', ME.message);
end

Headers = {'RaceID','Nro','Nimi','Valmentaja','Current_Ohjastaja','Ika','Sukupuoli','Is_Suomenhevonen','Kengat_Etu','Kengat_Taka',...
            'Kengat_etu_changed','Kengat_taka_changed','Current_Special_Cart','Scratched','Current_Distance','Current_Is_Auto','Current_Start_Date','Peli_pros','Voitto_pros','Ennatys_nro','Is_Auto_Record',...
            'Hist_PVM','Ohjastaja','Rata','Matka','RataNro','Km_aika','Hist_kengat_etu','Hist_kengat_taka','Hist_Special_Cart','Hist_Is_Auto','Laukka','Hylatty','Keskeytys','Track_Condition','Hist_Sij','Kerroin','Palkinto','SIJOITUS'};

T = cell2table(KaikkiData, 'VariableNames', Headers);
writetable(T, sprintf('Ravit_%s_lahto%d.csv', targetCardId, haluttuLahto), 'Delimiter', ';');