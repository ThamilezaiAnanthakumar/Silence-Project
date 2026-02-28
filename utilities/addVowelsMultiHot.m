function T = addVowelsMultiHot(T)
    % Add multi-hot encoding columns for vowels (NO NONE)
    n = height(T);
    T.isA = false(n, 1);
    T.isE = false(n, 1);
    T.isI = false(n, 1);
    T.isO = false(n, 1);
    T.isU = false(n, 1);
    
    for i = 1:n
        v = upper(char(T.Vowels(i)));  % Convert to char and uppercase
        if contains(v, 'A'), T.isA(i) = true; end
        if contains(v, 'E'), T.isE(i) = true; end
        if contains(v, 'I'), T.isI(i) = true; end
        if contains(v, 'O'), T.isO(i) = true; end
        if contains(v, 'U'), T.isU(i) = true; end
    end
    
    fprintf('Multi-hot encoding added:\n');
    fprintf('  isA: %d samples\n', sum(T.isA));
    fprintf('  isE: %d samples\n', sum(T.isE));
    fprintf('  isI: %d samples\n', sum(T.isI));
    fprintf('  isO: %d samples\n', sum(T.isO));
    fprintf('  isU: %d samples\n', sum(T.isU));
end