import React from 'react';
import { SUPPORTED_LANGUAGES } from '../types';

interface LanguageSelectorProps {
  label: string;
  value: string;
  onChange: (lang: string) => void;
  disabled?: boolean;
  excludeLang?: string;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  label,
  value,
  onChange,
  disabled = false,
  excludeLang,
}) => {
  return (
    <div className="language-selector">
      <label className="language-label">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="language-select"
      >
        {SUPPORTED_LANGUAGES.filter((l) => l.code !== excludeLang).map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.flag} {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
};
