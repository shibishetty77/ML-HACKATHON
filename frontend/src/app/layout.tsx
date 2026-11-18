import type { Metadata } from 'next';
import './globals.css';
import { AppProvider } from '@/context/AppContext';
import { Sidebar } from '@/components/layout/Sidebar';
import { TopNavbar } from '@/components/layout/TopNavbar';

export const metadata: Metadata = {
  title: 'PathSnap — Distributed Map-Matching & Route Reconstruction',
  description:
    'Real-time GPS trajectory map matching and route reconstruction with XGBoost & HMM Viterbi decoding.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen antialiased flex flex-row selection:bg-cyan-500 selection:text-black">
        <AppProvider>
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0">
            <TopNavbar />
            <main className="flex-1 p-6 overflow-y-auto">{children}</main>
          </div>
        </AppProvider>
      </body>
    </html>
  );
}
