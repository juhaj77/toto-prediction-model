%% TOTO-HISTORIA-AUTOMAATIO (Versio 2026 - 39 COLUMNS SYNC)
clearvars; clc;

% --- SETTINGS ---
startDate = '2026-02-22'; 
endDate = '2026-02-24';   
pvmLista = datestr(datetime(startDate):datetime(endDate), 'yyyy-mm-dd');
options = weboptions('ContentType', 'text', 'UserAgent', 'Mozilla/5.0', 'Timeout', 20, 'CharacterEncoding', 'UTF-8');

% JÄREÄ PUHDISTUS: Vaihtaa ä/ö/å -> a/o/a ja poistaa erikoismerkit
puhdista = @(str) regexprep(strrep(strrep(strrep(strrep(strrep(strrep(str, ...
    char(228), 'a'), char(246), 'o'), char(196), 'A'), char(214), 'O'), ...
    char(229), 'a'), char(197), 'A'), ...
    '[^a-zA-Z0-9\s\-\.\:]', '');

KaikkiData = {};

for d = 1:size(pvmLista, 1)
    tarkistusPvm = strtrim(pvmLista(d, :));
    fprintf('Day: %s\n', tarkistusPvm);
    
    try
        cRawText = webread(['https://www.veikkaus.fi/api/toto-info/v1/cards/date/', tarkistusPvm], options);
        cardIds = regexp(cRawText, '"cardId":(\d+)', 'tokens');
        if isempty(cardIds), continue; end

        for ci = 1:length(cardIds)
            cId = cardIds{ci}{1};
            rRaw = webread(sprintf('https://www.veikkaus.fi/api/toto-info/v1/card/%s/races', cId), options);
            raceBlocks = regexp(rRaw, '\{"raceId":(\d+),.*?"distance":(\d+),.*?"breed":"(.*?)".*?"startType":"(.*?)"', 'tokens');
            
            resultMatches = regexp(rRaw, '"raceId":(\d+),.*?"toteResultString":"([\d-]+)"', 'tokens');
            resMap = containers.Map();
            for k = 1:length(resultMatches)
                resMap(resultMatches{k}{1}) = resultMatches{k}{2};
            end

            for rb = 1:length(raceBlocks)
                vId = char(raceBlocks{rb}{1});
                currDist = str2double(raceBlocks{rb}{2});
                breedChar = char(raceBlocks{rb}{3});
                sTypeText = char(raceBlocks{rb}{4});
                
                isSH = double(strcmp(breedChar, 'K')); 
                currentIsAuto = double(strcmp(sTypeText, 'CAR_START')); 
                currentDate = puhdista(tarkistusPvm);
                
                tulosOsat = {};
                if isKey(resMap, vId), tulosOsat = strsplit(resMap(vId), '-'); end

                hRaw = webread(sprintf('https://www.veikkaus.fi/api/toto-info/v1/race/%s/runners', vId), options);
                hBlocks = strsplit(hRaw, '{"runnerId"'); hBlocks(1) = []; 

                for i = 1:length(hBlocks)
                    block = hBlocks{i};
                    nro = str2double(char(regexp(block, '"startNumber":(\d+)', 'tokens', 'once')));
                    nimi = puhdista(char(regexp(block, '"horseName":"(.*?)"', 'tokens', 'once')));
                    vNimi = puhdista(char(regexp(block, '"coachName":"(.*?)"', 'tokens', 'once')));
                    if isempty(vNimi), vNimi = 'Unknown'; end
                    
                    currOhjNimi = puhdista(char(regexp(block, '"driverName":"(.*?)"', 'tokens', 'once')));
                    if isempty(currOhjNimi), currOhjNimi = 'Unknown'; end
                    
                    ika = str2double(char(regexp(block, '"horseAge":(\d+)', 'tokens', 'once')));
                    spRaw = char(regexp(block, '"gender":"(.*?)"', 'tokens', 'once'));
                    spNum = 2; if strcmp(spRaw, 'TAMMA'), spNum = 1; elseif strcmp(spRaw, 'ORI'), spNum = 3; end
                    
                    palkit = strsplit(block, '{"priorStartId"');
                    nykyhetkiBlock = palkit{1}; 

                    % --- UUDET NYKYHETKI FEATURET ---
                    % Poimitaan nykyhetken kengitystieto tekstinä
                    k_etu = char(regexp(nykyhetkiBlock, '"frontShoes":"(.*?)"', 'tokens', 'once'));
                    if isempty(k_etu), k_etu = 'UNKNOWN'; end

                    k_taka = char(regexp(nykyhetkiBlock, '"rearShoes":"(.*?)"', 'tokens', 'once'));
                    if isempty(k_taka), k_taka = 'UNKNOWN'; end

                    k_etu_ch = double(~isempty(strfind(nykyhetkiBlock, '"frontShoesChanged":true')));
                    k_taka_ch = double(~isempty(strfind(nykyhetkiBlock, '"rearShoesChanged":true')));
                    
                    % Special Cart ja Scratched nykyhetkestä
                    curr_spec_cart = char(regexp(nykyhetkiBlock, '"specialCart":"(.*?)"', 'tokens', 'once'));
                    if isempty(curr_spec_cart), curr_spec_cart = 'UNKNOWN'; end
                    is_scratched = double(~isempty(strfind(nykyhetkiBlock, '"scratched":true')));

                    eMatch = regexp(block, '"(handicapRaceRecord|mobileStartRecord|vaultStartRecord)":"([\d,]+)[a-z]*"', 'tokens', 'once');
                    ennatys = 0; isAutoRec = 0;
                    if ~isempty(eMatch)
                        ennatys = str2double(strrep(eMatch{2}, ',', '.'));
                        isAutoRec = ~isempty(strfind(eMatch{1}, 'mobile'));
                    end

                    peliP = 0; pMatch = regexp(block, '"percentage":(\d+)', 'tokens', 'once');
                    if ~isempty(pMatch), peliP = str2double(pMatch{1}) / 100; end
                    
                    voittoP = 0;
                    statBlock = regexp(block, '"total":\{"year":"total",.*?"starts":(\d+),.*?"position1":(\d+)', 'tokens', 'once');
                    if ~isempty(statBlock)
                        sCount = str2double(statBlock{1});
                        wCount = str2double(statBlock{2});
                        if sCount > 0
                            voittoP = round((wCount / sCount) * 100, 2); 
                        end
                    end

                    toteutunutSij = 0;
                    if ~isempty(tulosOsat)
                        for s = 1:min(3, length(tulosOsat))
                            if strcmp(tulosOsat{s}, num2str(nro)), toteutunutSij = s; break; end
                        end
                    end

                    starts = strsplit(block, '{"priorStartId"'); starts(1) = []; 
                    if isempty(starts)
                        % 39 saraketta (tyhjä historia)
                        KaikkiData(end+1, 1:39) = {vId, nro, nimi, vNimi, currOhjNimi, ika, spNum, isSH, k_etu, k_taka, ...
                                   k_etu_ch, k_taka_ch, curr_spec_cart, is_scratched, currDist, currentIsAuto, currentDate, peliP, voittoP, ennatys, isAutoRec, ...
                                   '', '', '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '', 10, 0, 0, toteutunutSij};
                    else
                        for s = 1:length(starts)
                            sB = starts{s};
                            oNimi = puhdista(char(regexp(sB, '"driverFullName":"(.*?)"', 'tokens', 'once')));
                            rNimi = puhdista(char(regexp(sB, '"trackCode":"(.*?)"', 'tokens', 'once')));
                            
                            % Radan kunto historiasta
                            tCond = puhdista(char(regexp(sB, '"trackCondition":"(.*?)"', 'tokens', 'once')));
                            if isempty(tCond), tCond = 'unknown'; end
                            
                            kmRaw = char(regexp(sB, '"kmTime":"(.*?)"', 'tokens', 'once'));
                            histIsAuto = double(~isempty(strfind(kmRaw, 'a'))); 
                            isLaukka = double(~isempty(strfind(kmRaw, 'x')));
                            kmNumStr = regexp(kmRaw, '[\d,]+', 'match', 'once');
                            kmNum = str2double(strrep(kmNumStr, ',', '.')); if isnan(kmNum), kmNum = 0; end

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

                            % 39 saraketta (täysi historia)
                            KaikkiData(end+1, 1:39) = {vId, nro, nimi, vNimi, currOhjNimi, ika, spNum, isSH, k_etu, k_taka, ...
                                       k_etu_ch, k_taka_ch, curr_spec_cart, is_scratched, currDist, currentIsAuto, currentDate, peliP, voittoP, ennatys, isAutoRec, ...
                                       puhdista(char(regexp(sB, '"shortMeetDate":"(.*?)"', 'tokens', 'once'))), ...
                                       oNimi, rNimi, ...
                                       str2double(char(regexp(sB, '"distance":(\d+)', 'tokens', 'once'))), ...
                                       str2double(char(regexp(sB, '"startTrack":(\d+)', 'tokens', 'once'))), ...
                                       kmNum, h_k_etu, h_k_taka, h_spec_cart, histIsAuto, isLaukka, isHyl, isKesk, tCond, sijNum, ...
                                       str2double(char(regexp(sB, '"winOdd":"(\d+)"', 'tokens', 'once')))/10, ...
                                       str2double(char(regexp(sB, '"firstPrize":(\d+)', 'tokens', 'once')))/100, ...
                                       toteutunutSij};
                        end
                    end
                end 
            end 
        end
    catch ME
        fprintf('Error line %d: %s\n', ME.stack(1).line, ME.message);
    end
end

% --- SAVE ---
if ~isempty(KaikkiData)
    Headers = {'RaceID','Nro','Nimi','Valmentaja','Current_Ohjastaja','Ika','Sukupuoli','Is_Suomenhevonen','Kengat_Etu','Kengat_Taka',...
                'Kengat_etu_changed','Kengat_taka_changed','Current_Special_Cart','Scratched','Current_Distance','Current_Is_Auto','Current_Start_Date','Peli_pros','Voitto_pros','Ennatys_nro','Is_Auto_Record',...
                'Hist_PVM','Ohjastaja','Rata','Matka','RataNro','Km_aika','Hist_kengat_etu','Hist_kengat_taka','Hist_Special_Cart','Hist_Is_Auto','Laukka','Hylatty','Keskeytys','Track_Condition','Hist_Sij','Kerroin','Palkinto','SIJOITUS'};
    T = cell2table(KaikkiData, 'VariableNames', Headers);
    T.RaceID = cellstr(T.RaceID);
    writetable(T, 'Ravit_Opetus_Data_22-24.csv', 'Delimiter', ';', 'QuoteStrings', true);
    fprintf('Valmis! Tallennettu %d rivia.\n', size(KaikkiData,1));
end